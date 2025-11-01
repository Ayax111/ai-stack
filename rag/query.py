# ~/ai-stack/rag/query.py
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from dotenv import load_dotenv
from llm_client import chat  # cliente OpenAI-compatible hacia LM Studio
from sqlalchemy import create_engine, text

# Cargar .env (ruta explícita, override=True)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# =======================
# Configuración
# =======================
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("Falta DATABASE_URL en .env (o no se pudo cargar).")

TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))
DEBUG = os.getenv("DEBUG_RETRIEVE", "0") == "1"
MAX_CTX = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))

# Diversificación por documento
MAX_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "2"))
ROUND_ROBIN = os.getenv("ROUND_ROBIN_PER_DOC", "1") == "1"

# Reranker
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "0") == "1"
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "20"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "16"))

# Backend de embeddings (consulta)
BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()

# =======================
# Embeddings (consulta)
# =======================
if BACKEND == "sentence-transformers":
    from sentence_transformers import SentenceTransformer

    _enc = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))

    def embed(q: str) -> List[float]:
        return _enc.encode([q], normalize_embeddings=True)[0].tolist()

elif BACKEND == "lmstudio":
    import httpx
    from LMStudioManager import (
        DEFAULT_CHAT_PORT,
        DEFAULT_EMBED_PORT,
        DEFAULT_HOST,
        LMStudioManager,
        normalize_lmstudio_url,
    )

    EMB_BASE_ENV = os.getenv("EMBEDDING_BASE_URL")
    EMB_HOST_ENV = os.getenv("EMBEDDING_HOST")
    EMB_PORT_ENV = os.getenv("EMBEDDING_PORT")
    LLM_BASE_ENV = os.getenv("LLM_BASE_URL")

    parsed_fb = None
    fallback_host = EMB_HOST_ENV
    fallback_port = None
    if LLM_BASE_ENV:
        parsed_fb = urlparse(LLM_BASE_ENV if "://" in LLM_BASE_ENV else f"http://{LLM_BASE_ENV}")
        fallback_host = fallback_host or parsed_fb.hostname
        fallback_port = parsed_fb.port
    fallback_host = fallback_host or DEFAULT_HOST
    fallback_port = fallback_port or DEFAULT_CHAT_PORT

    if EMB_BASE_ENV:
        primary_base = EMB_BASE_ENV
    elif EMB_HOST_ENV or EMB_PORT_ENV:
        port_for_primary = int(EMB_PORT_ENV) if EMB_PORT_ENV else DEFAULT_EMBED_PORT
        primary_base = f"http://{fallback_host}:{port_for_primary}/v1"
    else:
        primary_base = None

    override_port = None
    if EMB_PORT_ENV:
        override_port = int(EMB_PORT_ENV)
    elif EMB_BASE_ENV:
        parsed_primary = urlparse(
            EMB_BASE_ENV if "://" in EMB_BASE_ENV else f"http://{EMB_BASE_ENV}"
        )
        primary_host = parsed_primary.hostname or fallback_host
        primary_port = parsed_primary.port or fallback_port
        if primary_host == fallback_host and primary_port == fallback_port:
            override_port = DEFAULT_EMBED_PORT
    else:
        override_port = DEFAULT_EMBED_PORT

    EMB_BASE = normalize_lmstudio_url(
        primary_base,
        fallback=LLM_BASE_ENV,
        default_host=fallback_host,
        default_port=DEFAULT_EMBED_PORT,
        override_port=override_port,
        override_host=EMB_HOST_ENV,
    )
    EMB_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "lm-studio"))
    EMB_MODEL = os.getenv("EMBEDDING_MODEL")
    if not EMB_MODEL:
        raise SystemExit("Con EMBEDDING_BACKEND=lmstudio, define EMBEDDING_MODEL en .env")
    EMB_ENDPOINT = (os.getenv("EMBEDDING_ENDPOINT") or "embeddings").strip("/")
    EMB_URL = f"{EMB_BASE}/{EMB_ENDPOINT}"
    TIMEOUT = httpx.Timeout(connect=5, read=120, write=30, pool=5)
    HEADERS = {"Authorization": f"Bearer {EMB_KEY}"}
    _embedding_manager = LMStudioManager(base_url=EMB_BASE or None, api_key=EMB_KEY)
    _embedding_ready = {"value": False}

    def _ensure_embedding_ready():
        if _embedding_ready["value"]:
            return
        ok = _embedding_manager.ensure_model_loaded(EMB_MODEL, wait_time=30)
        if not ok:
            raise RuntimeError(
                f"No se pudo cargar el modelo de embeddings {EMB_MODEL} en LM Studio."
            )
        _embedding_ready["value"] = True

    def embed(q: str) -> List[float]:
        _ensure_embedding_ready()
        payload = {"model": EMB_MODEL, "input": [q]}
        r = httpx.post(EMB_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404 and EMB_ENDPOINT == "embeddings":
            r = httpx.post(f"{EMB_BASE}/embedding", json=payload, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict) and "data" in js:
            return js["data"][0]["embedding"]
        if isinstance(js, dict) and "embedding" in js:
            return js["embedding"]
        raise RuntimeError(f"Respuesta inesperada del servidor de embeddings: {js}")
else:
    raise SystemExit(f"EMBEDDING_BACKEND no soportado: {BACKEND}")

# =======================
# DB
# =======================
engine = create_engine(DB_URL, future=True)


def retrieve(query: str, limit: int) -> List[Dict]:
    """Devuelve los 'limit' candidatos más cercanos por distancia vectorial (+threshold)."""
    qvec = embed(query)
    qlit = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT doc_id, chunk_id, content,
                   1 - (embedding <=> (:qvec)::vector) AS score
            FROM documents
            ORDER BY embedding <=> (:qvec)::vector
            LIMIT :limit
        """
                ),
                {"qvec": qlit, "limit": limit},
            )
            .mappings()
            .all()
        )
    return [r for r in rows if r["score"] is None or r["score"] >= SCORE_THRESHOLD]


# =======================
# Filtrado y diversidad
# =======================
def _filter_and_diversify(rows: List[Dict]) -> List[Dict]:
    """Aplica MAX_PER_DOC; opcionalmente round-robin; corta a TOP_K."""
    if not rows:
        return []

    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(r["doc_id"], []).append(r)

    # Limitar por documento respetando el orden por similitud
    for doc in groups:
        groups[doc] = groups[doc][:MAX_PER_DOC]

    if ROUND_ROBIN:
        queues = [deque(groups[doc]) for doc in sorted(groups.keys())]
        mixed: List[Dict] = []
        added = 0
        while queues and added < TOP_K:
            new_queues = []
            for q in queues:
                if q and added < TOP_K:
                    mixed.append(q.popleft())
                    added += 1
                if q:
                    new_queues.append(q)
            queues = new_queues
        return mixed
    else:
        mixed: List[Dict] = []
        for doc in groups:
            mixed.extend(groups[doc])
            if len(mixed) >= TOP_K:
                break
        return mixed[:TOP_K]


def build_context(selected: List[Dict], all_rows: List[Dict]) -> str:
    """Construye el contexto con límite MAX_CTX y DEBUG opcional."""
    if DEBUG:
        print("\n== Pasajes recuperados (raw) ==")
        for r in all_rows:
            preview = (r["content"] or "").replace("\n", " ")[:90]
            print(f"[{r['doc_id']}#{r['chunk_id']}] score={r['score']:.3f} → {preview}...")
        print("\n== Pasajes seleccionados (tras límites por doc / threshold) ==")
        for r in selected:
            preview = (r["content"] or "").replace("\n", " ")[:90]
            print(f"✔ [{r['doc_id']}#{r['chunk_id']}] score={r['score']:.3f} → {preview}...")

    parts: List[str] = []
    total = 0
    for r in selected:
        prefix = f"[{r['doc_id']}#{r['chunk_id']}] "
        text = (r["content"] or "").strip()
        piece = prefix + text
        if total + len(piece) + 2 > MAX_CTX:
            break
        parts.append(piece)
        total += len(piece) + 2
    return "\n\n".join(parts)


def format_sources(selected: List[Dict]) -> str:
    grouped: Dict[str, List[int]] = {}
    for r in selected:
        grouped.setdefault(r["doc_id"], []).append(r["chunk_id"])
    if not grouped:
        return "Fuentes: (sin pasajes seleccionados)"
    lines: List[str] = []
    for doc, chunks in grouped.items():
        uniq = sorted(set(chunks))
        lines.append(f"- {doc}  (chunks: {', '.join(map(str, uniq))})")
    return "Fuentes:\n" + "\n".join(lines)


# =======================
# Reranker (opcional)
# =======================
def maybe_rerank(query: str, rows: List[Dict]) -> List[Dict]:
    """Aplica cross-encoder si está habilitado; devuelve rows reordenados."""
    if not RERANKER_ENABLED or not rows:
        return rows
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(RERANKER_MODEL)
        pairs = [(query, (r["content"] or "")) for r in rows]
        scores = model.predict(pairs, batch_size=RERANKER_BATCH_SIZE)
        augmented = list(zip(scores, rows))
        augmented.sort(key=lambda x: float(x[0]), reverse=True)
        return [r for s, r in augmented]
    except Exception as e:
        if DEBUG:
            print(f"[WARN] Fallo al usar el reranker: {e}")
        return rows


# =======================
# Orquestación de la respuesta
# =======================
def answer(query: str, k: int) -> str:
    # nº de candidatos a pedir al retriever (si hay reranker, pedimos más)
    cand = RERANKER_CANDIDATES if RERANKER_ENABLED else k

    # recuperar más candidatos
    all_rows = retrieve(query, limit=cand)

    # rerank opcional (reordena por cross-encoder)
    all_rows = maybe_rerank(query, all_rows)

    # filtros, diversidad y corte a TOP_K
    selected = _filter_and_diversify(all_rows)

    context = build_context(selected, all_rows)
    if not context:
        return "No tengo datos suficientes en el contexto para responder con confianza."

    system = (
        "Eres un asistente preciso. Responde SOLO con la información del CONTEXTO. "
        "Si falta información, di explícitamente: 'No tengo datos suficientes en el contexto'. "
        "Cita usando el formato [doc#chunk]. Sé conciso."
    )
    user = f"Pregunta: {query}\n\n--- CONTEXTO ---\n{context}\n\nResponde:"

    reply = chat(system, user, temperature=0.2)
    return reply.strip() + "\n\n" + format_sources(selected)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "¿Qué es MCP y para qué sirve?"
    print(answer(q, k=TOP_K))

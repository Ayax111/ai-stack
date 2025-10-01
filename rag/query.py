# ~/ai-stack/rag/query.py
from __future__ import annotations

import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from llm_client import chat  # cliente OpenAI-compatible hacia LM Studio
from sqlalchemy import create_engine, text

# =======================
# Carga de configuración
# =======================
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("Falta DATABASE_URL en .env")

# Retrieve / debug
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

# Backend de embeddings (consulta)
BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()

# =======================
# Embeddings (consulta)
# =======================
if BACKEND == "sentence-transformers":
    from sentence_transformers import SentenceTransformer

    _enc = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))

    def embed(q: str) -> List[float]:
        # normalizamos para trabajar bien con cosine
        return _enc.encode([q], normalize_embeddings=True)[0].tolist()

elif BACKEND == "lmstudio":
    import httpx

    EMB_BASE = os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "")).rstrip("/")
    EMB_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "lm-studio"))
    EMB_MODEL = os.getenv("EMBEDDING_MODEL")
    if not EMB_MODEL:
        raise SystemExit("Con EMBEDDING_BACKEND=lmstudio, define EMBEDDING_MODEL en .env")
    EMB_ENDPOINT = (os.getenv("EMBEDDING_ENDPOINT") or "embeddings").strip("/")
    EMB_URL = f"{EMB_BASE}/{EMB_ENDPOINT}"
    TIMEOUT = httpx.Timeout(connect=5, read=120, write=30, pool=5)
    HEADERS = {"Authorization": f"Bearer {EMB_KEY}"}

    def embed(q: str) -> List[float]:
        payload = {"model": EMB_MODEL, "input": [q]}
        r = httpx.post(EMB_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404 and EMB_ENDPOINT == "embeddings":
            # fallback por si alguna build expone 'embedding' en singular
            r = httpx.post(f"{EMB_BASE}/embedding", json=payload, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["embedding"]
else:
    raise SystemExit(f"EMBEDDING_BACKEND no soportado: {BACKEND}")

# =======================
# DB
# =======================
engine = create_engine(DB_URL, future=True)


def retrieve(query: str, limit: int) -> List[Dict]:
    """Devuelve los 'limit' candidatos más cercanos por distancia vectorial."""
    qvec = embed(query)
    qlit = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text("""
            SELECT doc_id, chunk_id, content,
                   1 - (embedding <=> (:qvec)::vector) AS score
            FROM documents
            ORDER BY embedding <=> (:qvec)::vector
            LIMIT :limit
        """),
                {"qvec": qlit, "limit": limit},
            )
            .mappings()
            .all()
        )
    return rows


# =======================
# Filtrado y diversidad
# =======================
def _filter_and_diversify(rows: List[Dict]) -> List[Dict]:
    """Aplica SCORE_THRESHOLD y MAX_PER_DOC; opcionalmente round-robin entre docs."""
    eligible = [r for r in rows if r["score"] is None or r["score"] >= SCORE_THRESHOLD]
    if not eligible:
        return []

    groups = defaultdict(list)
    for r in eligible:
        groups[r["doc_id"]].append(r)

    # Limitar por documento respetando el orden por similitud
    for doc in groups:
        groups[doc] = groups[doc][:MAX_PER_DOC]

    if ROUND_ROBIN:
        # Alterna entre documentos para más cobertura
        queues = [deque(groups[doc]) for doc in sorted(groups.keys())]
        mixed, added = [], 0
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
        # Secuencial por grupos
        mixed: List[Dict] = []
        for doc in groups:
            mixed.extend(groups[doc])
            if len(mixed) >= TOP_K:
                break
        return mixed[:TOP_K]


def build_context(selected: List[Dict], all_rows: List[Dict]) -> str:
    """Construye el contexto con límite MAX_CTX y, si DEBUG, imprime detalles."""
    if DEBUG:
        print("\n== Pasajes recuperados (raw) ==")
        for r in all_rows:
            preview = (r["content"] or "").replace("\n", " ")[:90]
            print(f"[{r['doc_id']}#{r['chunk_id']}] score={r['score']:.3f} → {preview}...")
    if DEBUG:
        print("\n== Pasajes seleccionados (tras límites por doc / threshold) ==")
        for r in selected:
            preview = (r["content"] or "").replace("\n", " ")[:90]
            print(f"✔ [{r['doc_id']}#{r['chunk_id']}] score={r['score']:.3f} → {preview}...")

    parts, total = [], 0
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
    """Lista de fuentes agrupadas por documento."""
    grouped = defaultdict(list)
    for r in selected:
        grouped[r["doc_id"]].append(r["chunk_id"])
    if not grouped:
        return "Fuentes: (sin pasajes seleccionados)"
    lines = []
    for doc, chunks in grouped.items():
        uniq = sorted(set(chunks))
        lines.append(f"- {doc}  (chunks: {', '.join(map(str, uniq))})")
    return "Fuentes:\n" + "\n".join(lines)


# =======================
# Reranker (opcional)
# =======================
def maybe_rerank(query: str, rows: List[Dict], candidates: int) -> List[Dict]:
    """Aplica cross-encoder si está habilitado; devuelve rows reordenados."""
    if not RERANKER_ENABLED or not rows:
        return rows
    try:
        from reranker import rerank as crossenc_rerank
    except Exception as e:
        if DEBUG:
            print(f"[WARN] No se pudo importar el reranker: {e}")
        return rows
    return crossenc_rerank(query, rows, top_k=candidates)


# =======================
# Orquestación de la respuesta
# =======================
def answer(query: str, k: int) -> str:
    # nº de candidatos a pedir al retriever (si hay reranker, pedimos más)
    cand = RERANKER_CANDIDATES if RERANKER_ENABLED else k

    # recuperar
    all_rows = retrieve(query, limit=cand)

    # rerank opcional (reordena por cross-encoder)
    all_rows = maybe_rerank(query, all_rows, candidates=cand)

    # filtros y diversidad
    selected = _filter_and_diversify(all_rows)

    # construir contexto
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


# =======================
# CLI
# =======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Consulta RAG sobre tus documentos")
    parser.add_argument("question", nargs="*", help="Texto de la pregunta")
    parser.add_argument("--k", type=int, help="Top-K final (por defecto: TOP_K de .env)")
    parser.add_argument("--threshold", type=float, help="Umbral de score 0..1 (SCORE_THRESHOLD)")
    parser.add_argument(
        "--max-per-doc", type=int, help="Máximo de chunks por documento (MAX_CHUNKS_PER_DOC)"
    )
    parser.add_argument(
        "--no-round-robin", action="store_true", help="Desactivar round-robin entre docs"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Forzar activar reranker (ignora .env)"
    )
    parser.add_argument("--no-rerank", action="store_true", help="Forzar desactivar reranker")
    parser.add_argument(
        "--max-ctx", type=int, help="Límite de caracteres del contexto (MAX_CONTEXT_CHARS)"
    )
    args = parser.parse_args()

    # Overrides de runtime (si se pasan)
    if args.k is not None:
        TOP_K = args.k
    if args.threshold is not None:
        SCORE_THRESHOLD = args.threshold
    if args.max_per_doc is not None:
        MAX_PER_DOC = args.max_per_doc
    if args.no_round_robin:
        ROUND_ROBIN = False
    if args.max_ctx is not None:
        MAX_CTX = args.max_ctx
    if args.rerank:
        RERANKER_ENABLED = True
    if args.no_rerank:
        RERANKER_ENABLED = False

    q = " ".join(args.question) or "¿Qué es MCP y para qué sirve?"
    print(answer(q, k=TOP_K))

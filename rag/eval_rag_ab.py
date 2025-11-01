from __future__ import annotations

import csv
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Carga .env local del directorio (override=True para refrescar cambios)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("Falta DATABASE_URL en .env")

# Parámetros de retrieve / control
TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))

# Diversidad por documento (alineado con query.py)
MAX_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "2"))
ROUND_ROBIN = os.getenv("ROUND_ROBIN_PER_DOC", "1") == "1"

# Reranker
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "20"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "16"))

# Backend de embeddings para la consulta
BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()

# --------- Embeddings (consulta) ----------
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
    EMB_MODEL = os.getenv("EMBEDDING_MODEL") or ""
    EMB_ENDPOINT = (os.getenv("EMBEDDING_ENDPOINT") or "embeddings").strip("/")
    EMB_URL = f"{EMB_BASE}/{EMB_ENDPOINT}"
    TIMEOUT = httpx.Timeout(connect=5, read=120, write=30, pool=5)
    HEADERS = {"Authorization": f"Bearer {EMB_KEY}"}
    _embedding_manager = LMStudioManager(base_url=EMB_BASE, api_key=EMB_KEY)
    _embedding_ready = {"value": False}

    def _ensure_embedding_ready() -> None:
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
        # Fallback si /embeddings no existe y solo está /embedding
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

# --------- DB ----------
engine = create_engine(DB_URL, future=True)


def retrieve(query: str, k: int) -> List[Dict]:
    """Devuelve k candidatos ordenados por similitud vectorial (con threshold)."""
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
            LIMIT :k
        """
                ),
                {"qvec": qlit, "k": k},
            )
            .mappings()
            .all()
        )
    return [r for r in rows if r["score"] is None or r["score"] >= SCORE_THRESHOLD]


# --------- Reranker ----------
_cross = None


def _load_reranker():
    global _cross
    if _cross is None:
        from sentence_transformers import CrossEncoder

        _cross = CrossEncoder(RERANKER_MODEL)
    return _cross


def rerank(query: str, rows: List[Dict]) -> List[Dict]:
    if not rows:
        return rows
    try:
        model = _load_reranker()
    except Exception as e:
        print(f"[WARN] Reranker no disponible ({e}); usando baseline.")
        return rows
    pairs = [(query, (r["content"] or "")) for r in rows]
    scores = model.predict(pairs, batch_size=RERANKER_BATCH_SIZE)
    augmented = list(zip(scores, rows))
    augmented.sort(key=lambda x: float(x[0]), reverse=True)
    return [r for s, r in augmented]


# --------- Diversidad por documento (igual que query.py) ----------
def _filter_and_diversify(rows: List[Dict], k: int) -> List[Dict]:
    if not rows:
        return []
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(r["doc_id"], []).append(r)

    # Limita por documento respetando orden
    for doc in groups:
        groups[doc] = groups[doc][:MAX_PER_DOC]

    if ROUND_ROBIN:
        queues = [deque(groups[doc]) for doc in sorted(groups.keys())]
        mixed: List[Dict] = []
        added = 0
        while queues and added < k:
            new_queues = []
            for q in queues:
                if q and added < k:
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
            if len(mixed) >= k:
                break
        return mixed[:k]


# --------- Métricas ----------
def contains_all(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return all(term.lower() in t for term in terms) if terms else True


def contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms) if terms else True


def evaluate_case(case: Dict, k: int, variant: str) -> Dict:
    q = case["query"]
    must = case.get("must_have", [])
    should = case.get("should_have", [])

    # Recupera más candidatos si vamos a rerankear
    cand = max(k, RERANKER_CANDIDATES if variant == "rerank" else k)

    t0 = time.perf_counter()
    rows = retrieve(q, cand)
    if variant == "rerank":
        rows = rerank(q, rows)
    # APLICAR DIVERSIDAD Y CORTAR A k (igual que en query.py)
    rows = _filter_and_diversify(rows, k)
    dt = time.perf_counter() - t0

    # contexto combinado
    context = "\n\n".join((r["content"] or "") for r in rows)

    # métricas
    rank_first_full = None
    for idx, r in enumerate(rows, start=1):
        if contains_all(r["content"] or "", must):
            rank_first_full = idx
            break

    hit_at_k = 1 if rank_first_full is not None else 0
    mrr = 1 / rank_first_full if rank_first_full else 0.0

    return {
        "id": case.get("id"),
        "variant": variant,
        "k": k,
        "n_rows": len(rows),
        "hit@k": hit_at_k,
        "mrr": round(mrr, 3),
        "must_ctx": int(contains_all(context, must)),
        "should_ctx": int(contains_any(context, should)),
        "lat_ms": int(dt * 1000),
    }


def main():
    tests_path = Path(__file__).with_name("tests.yaml")
    if not tests_path.exists():
        raise SystemExit(f"No existe {tests_path}. Crea rag/tests.yaml.")

    cases = yaml.safe_load(tests_path.read_text()) or []
    if not cases:
        raise SystemExit("tests.yaml está vacío.")

    out_dir = Path(__file__).with_name("out")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "eval_ab.csv"

    results: List[Dict] = []
    for case in cases:
        results.append(evaluate_case(case, TOP_K, "baseline"))
        results.append(evaluate_case(case, TOP_K, "rerank"))

    # agregados por variante
    def aggregate(rows: List[Dict]) -> Dict:
        n = len(rows)
        return {
            "hit@k": sum(r["hit@k"] for r in rows) / n,
            "mrr": sum(r["mrr"] for r in rows) / n,
            "must_ctx": sum(r["must_ctx"] for r in rows) / n,
            "should_ctx": sum(r["should_ctx"] for r in rows) / n,
            "lat_ms_avg": int(sum(r["lat_ms"] for r in rows) / n),
        }

    base_agg = aggregate([r for r in results if r["variant"] == "baseline"])
    rr_agg = aggregate([r for r in results if r["variant"] == "rerank"])

    print("\n# Agregados (baseline vs rerank)")
    for key in ["hit@k", "mrr", "must_ctx", "should_ctx", "lat_ms_avg"]:
        b = base_agg[key]
        r = rr_agg[key]
        delta = r - b
        if isinstance(b, float):
            print(f"{key:12}: baseline={b:.3f} | rerank={r:.3f} | Δ={delta:+.3f}")
        else:
            print(f"{key:12}: baseline={b} | rerank={r} | Δ={delta:+}")

    # guarda CSV
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV → {out_csv}")


if __name__ == "__main__":
    main()

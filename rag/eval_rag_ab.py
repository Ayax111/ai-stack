# rag/eval_rag_ab.py
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Carga .env local del directorio
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("Falta DATABASE_URL en .env")

TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "20"))

# Embeddings (consulta)
BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()
if BACKEND == "sentence-transformers":
    from sentence_transformers import SentenceTransformer

    _enc = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))

    def embed(q: str) -> List[float]:
        return _enc.encode([q], normalize_embeddings=True)[0].tolist()

elif BACKEND == "lmstudio":
    import httpx

    EMB_BASE = os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "")).rstrip("/")
    EMB_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "lm-studio"))
    EMB_MODEL = os.getenv("EMBEDDING_MODEL") or ""
    EMB_ENDPOINT = (os.getenv("EMBEDDING_ENDPOINT") or "embeddings").strip("/")
    EMB_URL = f"{EMB_BASE}/{EMB_ENDPOINT}"
    TIMEOUT = httpx.Timeout(connect=5, read=120, write=30, pool=5)
    HEADERS = {"Authorization": f"Bearer {EMB_KEY}"}

    def embed(q: str) -> List[float]:
        payload = {"model": EMB_MODEL, "input": [q]}
        r = httpx.post(EMB_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404 and EMB_ENDPOINT == "embeddings":
            r = httpx.post(f"{EMB_BASE}/embedding", json=payload, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
else:
    raise SystemExit(f"EMBEDDING_BACKEND no soportado: {BACKEND}")

# Reranker (lazy)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_cross = None


def _load_reranker():
    global _cross
    if _cross is None:
        from sentence_transformers import CrossEncoder

        _cross = CrossEncoder(RERANKER_MODEL)
    return _cross


engine = create_engine(DB_URL, future=True)


def retrieve(query: str, k: int) -> List[Dict]:
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


def rerank(query: str, rows: List[Dict]) -> List[Dict]:
    if not rows:
        return rows
    try:
        model = _load_reranker()
    except Exception as e:
        print(f"[WARN] Reranker no disponible ({e}); usando baseline.")
        return rows
    pairs = [(query, (r["content"] or "")) for r in rows]
    scores = model.predict(pairs)
    augmented = list(zip(scores, rows))
    augmented.sort(key=lambda x: float(x[0]), reverse=True)
    return [r for s, r in augmented]


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
    # nos quedamos con k tras (re)ordenar
    rows = rows[:k]
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
        raise SystemExit(f"No existe {tests_path}.")
    import yaml

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

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv
from sqlalchemy import text

# Carga .env del directorio actual
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("Falta DATABASE_URL en .env")

TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))

# Reutilizamos la misma función de embed que en query.py (sin importar todo el archivo)
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

engine = None


def get_engine():
    global engine
    if engine is None:
        from sqlalchemy import create_engine as _ce

        engine = _ce(DB_URL, future=True)
    return engine


def retrieve(query: str, k: int) -> List[Dict]:
    qvec = embed(query)
    qlit = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    with get_engine().begin() as conn:
        rows = (
            conn.execute(
                text("""
            SELECT doc_id, chunk_id, content,
                   1 - (embedding <=> (:qvec)::vector) AS score
            FROM documents
            ORDER BY embedding <=> (:qvec)::vector
            LIMIT :k
        """),
                {"qvec": qlit, "k": k},
            )
            .mappings()
            .all()
        )
    # aplica umbral básico (igual que en query.py)
    return [r for r in rows if r["score"] is None or r["score"] >= SCORE_THRESHOLD]


def contains_all(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return all(term.lower() in t for term in terms)


def contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms) if terms else True


def evaluate_case(case: Dict, k: int) -> Dict:
    q = case["query"]
    must = case.get("must_have", [])
    should = case.get("should_have", [])

    t0 = time.perf_counter()
    rows = retrieve(q, k)
    dt = time.perf_counter() - t0

    # métricas de retrieve (sin “ground truth” de doc, usamos heuris. de contenido)
    # Contexto combinado (lo que enviaríamos al LLM)
    context = "\n\n".join((r["content"] or "") for r in rows)
    # Cobertura “must” en contexto
    must_in_context = contains_all(context, must)
    # Cobertura “should” en contexto
    should_in_context = contains_any(context, should)

    # Rank del primer chunk que contiene TODOS los must (si existe)
    rank_first_full = None
    for idx, r in enumerate(rows, start=1):
        if contains_all(r["content"] or "", must):
            rank_first_full = idx
            break

    hit_at_k = 1 if rank_first_full is not None else 0
    mrr = 1 / rank_first_full if rank_first_full else 0.0

    return {
        "id": case.get("id"),
        "k": k,
        "n_rows": len(rows),
        "hit@k": hit_at_k,
        "mrr": round(mrr, 3),
        "must_in_context": bool(must_in_context),
        "should_in_context": bool(should_in_context),
        "lat_ms": int(dt * 1000),
    }


def main():
    tests_path = Path(__file__).with_name("tests.yaml")
    if not tests_path.exists():
        raise SystemExit(f"No existe {tests_path}. Crea rag/tests.yaml")

    tests = yaml.safe_load(tests_path.read_text()) or []
    if not tests:
        raise SystemExit("tests.yaml está vacío.")

    k = int(os.getenv("TOP_K", "5"))
    results = [evaluate_case(tc, k) for tc in tests]

    # agregados
    agg = {
        "hit@k": sum(r["hit@k"] for r in results) / len(results),
        "mrr": sum(r["mrr"] for r in results) / len(results),
        "must_ctx": sum(1 for r in results if r["must_in_context"]) / len(results),
        "should_ctx": sum(1 for r in results if r["should_in_context"]) / len(results),
        "lat_ms_avg": int(sum(r["lat_ms"] for r in results) / len(results)),
    }

    # salida bonita sin dependencias; imprime tabla simple
    print("\n# Resultados por caso")
    print("id\tk\thit@k\tmrr\tmust_ctx\tshould_ctx\tlat_ms")
    for r in results:
        print(
            f"{r['id']}\t{r['k']}\t{r['hit@k']}\t{r['mrr']}\t{int(r['must_in_context'])}\t\t{int(r['should_in_context'])}\t\t{r['lat_ms']}"
        )

    print("\n# Agregados")
    for k_, v in agg.items():
        if isinstance(v, float):
            print(f"{k_}: {v:.3f}")
        else:
            print(f"{k_}: {v}")


if __name__ == "__main__":
    main()

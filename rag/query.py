# ~/ai-stack/rag/query.py
import os
from typing import List, Dict
from collections import defaultdict

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# usamos el mismo cliente de chat que ya tienes
from llm_client import chat

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("Falta DATABASE_URL en .env")

# --- Parámetros de retrieve/depuración desde .env ---
TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))
DEBUG = os.getenv("DEBUG_RETRIEVE", "0") == "1"
MAX_CTX = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))

# --- Backend de embeddings (igual que en ingest.py) ---
BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()

if BACKEND == "sentence-transformers":
    from sentence_transformers import SentenceTransformer
    _enc = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    def embed(q: str) -> List[float]:
        # normalizamos para cosine
        return _enc.encode([q], normalize_embeddings=True)[0].tolist()

elif BACKEND == "lmstudio":
    import httpx
    EMB_BASE = os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "")).rstrip("/")
    EMB_KEY  = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "lm-studio"))
    EMB_MODEL= os.getenv("EMBEDDING_MODEL")
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
            r = httpx.post(f"{EMB_BASE}/embedding", json=payload, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["embedding"]
else:
    raise SystemExit(f"EMBEDDING_BACKEND no soportado: {BACKEND}")

engine = create_engine(DB_URL, future=True)

def retrieve(query: str, k: int) -> List[Dict]:
    qvec = embed(query)
    qlit = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT doc_id, chunk_id, content,
                   1 - (embedding <=> (:qvec)::vector) AS score
            FROM documents
            ORDER BY embedding <=> (:qvec)::vector
            LIMIT :k
        """), {"qvec": qlit, "k": k}).mappings().all()
    return rows

def build_context(rows: List[Dict]) -> str:
    """Filtra por SCORE_THRESHOLD y limita a MAX_CTX caracteres."""
    kept = [r for r in rows if r["score"] is None or r["score"] >= SCORE_THRESHOLD]
    if DEBUG:
        print("\n== Pasajes recuperados ==")
        for r in rows:
            mark = "✔" if r in kept else "✖"
            preview = (r["content"] or "").replace("\n", " ")[:90]
            print(f"{mark} [{r['doc_id']}#{r['chunk_id']}] score={r['score']:.3f} → {preview}...")
    # Construye contexto respetando límite de caracteres
    ctx_parts = []
    total = 0
    for r in kept:
        prefix = f"[{r['doc_id']}#{r['chunk_id']}] "
        text = (r["content"] or "").strip()
        piece = prefix + text
        if total + len(piece) + 2 > MAX_CTX:
            break
        ctx_parts.append(piece)
        total += len(piece) + 2
    return "\n\n".join(ctx_parts)

def format_sources(rows: List[Dict]) -> str:
    """Agrupa citas por documento para mostrarlas ordenadas al final."""
    grouped = defaultdict(list)
    for r in rows:
        if r["score"] is not None and r["score"] >= SCORE_THRESHOLD:
            grouped[r["doc_id"]].append(r["chunk_id"])
    if not grouped:
        return "Fuentes: (sin pasajes por encima del umbral)"
    parts = []
    for doc, chunks in grouped.items():
        uniq = sorted(set(chunks))
        parts.append(f"- {doc}  (chunks: {', '.join(map(str, uniq))})")
    return "Fuentes:\n" + "\n".join(parts)

def answer(query: str, k: int) -> str:
    rows = retrieve(query, k=k)
    context = build_context(rows)
    if not context:
        return "No tengo datos suficientes en el contexto para responder con confianza."

    system = (
        "Eres un asistente preciso. Responde SOLO con la información del CONTEXTO. "
        "Si falta información, di explícitamente: 'No tengo datos suficientes en el contexto'. "
        "Cuando cites, usa el formato [doc#chunk]. Sé conciso y directo."
    )
    user = f"Pregunta: {query}\n\n--- CONTEXTO ---\n{context}\n\nResponde:"
    reply = chat(system, user, temperature=0.2)

    # Añade bloque de fuentes al final (ayuda a auditar)
    return reply.strip() + "\n\n" + format_sources(rows)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "¿Qué es MCP y para qué sirve?"
    print(answer(q, k=TOP_K))

# ~/ai-stack/rag/ingest.py
"""Pipeline que trocea documentos, calcula embeddings y los guarda en Postgres."""

import glob
import json
import os
import re
from datetime import datetime
from typing import List
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import create_engine, text

# Cargamos variables definidas en `.env` (Rutas, claves, configuración del modelo, etc.).
load_dotenv()

# ----------------- Configuración desde .env -----------------
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("Falta DATABASE_URL en .env")

# DATA_DIR puede ser absoluta o relativa al cwd
DATA_DIR = os.getenv("DATA_DIR", "data")
DATA_DIR = os.path.abspath(os.path.expanduser(DATA_DIR))

# Troceado (opcional, con valores por defecto)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Embeddings: backend y modelo
BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()

# Pool de conexiones hacia la base donde persistimos los embeddings.
engine = create_engine(DB_URL, future=True)

# ----------------- Backend de embeddings -----------------
if BACKEND == "sentence-transformers":
    from sentence_transformers import SentenceTransformer

    # Cargamos el modelo local elegido (por defecto BAAI/bge-m3).
    MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    _enc = SentenceTransformer(MODEL_NAME)
    EMB_DIM = _enc.get_sentence_embedding_dimension()

    def embed_batch(texts: List[str]) -> List[List[float]]:
        """Calcula embeddings normalizados de un lote de textos en local."""

        # Normalizamos para que el índice HNSW trabaje con coseno directamente.
        return _enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()

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
    _embedding_manager = LMStudioManager(base_url=EMB_BASE, api_key=EMB_KEY)
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

    def _post_embeddings(texts):
        """Realiza la petición HTTP que devuelve los embeddings para un lote de textos."""

        _ensure_embedding_ready()
        payload = {"model": EMB_MODEL, "input": texts}
        r = httpx.post(EMB_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404 and EMB_ENDPOINT == "embeddings":
            r = httpx.post(f"{EMB_BASE}/embedding", json=payload, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return [d["embedding"] for d in data["data"]]

    print(f"🔎 Embeddings LM Studio → {EMB_URL} | model={EMB_MODEL}")

    EMB_DIM_ENV = os.getenv("EMBEDDING_DIM")
    if EMB_DIM_ENV:
        EMB_DIM = int(EMB_DIM_ENV)
    else:
        probe = _post_embeddings(["hola"])
        EMB_DIM = len(probe[0])

    def embed_batch(texts: List[str]) -> List[List[float]]:
        """Envía el lote de textos al servidor remoto y devuelve los embeddings crudos."""

        return _post_embeddings(texts)

else:
    raise SystemExit(f"EMBEDDING_BACKEND no soportado: {BACKEND}")


# ----------------- Utilidades -----------------
def ensure_schema():
    """Crea la tabla documents con la dimensión correcta y un índice HNSW (cosine)."""

    # Operamos dentro de una transacción; se crea la extensión vector si no existe.
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        exists = conn.execute(text("SELECT to_regclass('public.documents')")).scalar()
        if exists:
            # Verifica dimensión existente
            typ = conn.execute(
                text("""
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                WHERE a.attrelid = 'documents'::regclass
                  AND a.attname = 'embedding' AND NOT a.attisdropped
            """)
            ).scalar()
            m = re.search(r"vector\((\d+)\)", typ or "")
            if m and int(m.group(1)) != EMB_DIM:
                count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
                raise SystemExit(
                    f"⚠️ La tabla 'documents' ya existe con dimensión {m.group(1)} "
                    f"y contiene {count} filas, pero tu modelo produce {EMB_DIM}.\n"
                    f"Elimina o migra la tabla antes de reingestar (p. ej., DROP TABLE documents;)."
                )
        else:
            # Si no existe la tabla, la creamos junto con el índice aproximado.
            conn.execute(
                text(f"""
                CREATE TABLE documents (
                  id BIGSERIAL PRIMARY KEY,
                  doc_id   TEXT NOT NULL,
                  chunk_id INT  NOT NULL,
                  content  TEXT NOT NULL,
                  metadata JSONB,
                  embedding VECTOR({EMB_DIM})
                );
            """)
            )
            conn.execute(
                text("""
                CREATE INDEX documents_embedding_hnsw
                ON documents USING hnsw (embedding vector_cosine_ops)
            """)
            )


def read_text(path: str) -> str:
    """Lee un archivo de texto o extrae el contenido de un PDF usando pypdf."""

    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split(text: str) -> List[str]:
    """Divide el texto en fragmentos solapados para mejorar el recall del RAG."""

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_text(text)


def to_vec_literal(v: List[float]) -> str:
    """Convierte una lista de floats a literal PostgreSQL compatible con pgvector."""

    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def upsert(doc_id: str, chunks: List[str]):
    """Inserta fragmentos y embeddings en la tabla (ignora duplicados exactos)."""

    with engine.begin() as conn:
        for i in range(0, len(chunks), 64):
            sub = chunks[i : i + 64]
            vecs = embed_batch(sub)
            now = datetime.utcnow().isoformat()
            rows = []
            for j, (txt, emb) in enumerate(zip(sub, vecs)):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": i + j,
                        "content": txt,
                        "metadata": json.dumps({"source": doc_id, "ingested_at": now}),
                        "embedding": to_vec_literal(emb),
                    }
                )
            conn.execute(
                text("""
                INSERT INTO documents (doc_id, chunk_id, content, metadata, embedding)
                VALUES (:doc_id, :chunk_id, :content, (:metadata)::jsonb, (:embedding)::vector)
                ON CONFLICT DO NOTHING
            """),
                rows,
            )


def collect_paths(root: str) -> List[str]:
    """Recupera todos los archivos de texto/markdown/pdf de forma recursiva."""

    if not os.path.isdir(root):
        raise SystemExit(f"La carpeta DATA_DIR no existe: {root}")
    pats = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        pats += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(pats)


def main():
    """Ejecuta la ingesta completa: preparar BD, leer archivos y guardar embeddings."""

    print(f"📂 Usando DATA_DIR: {DATA_DIR}")
    ensure_schema()
    paths = collect_paths(DATA_DIR)
    if not paths:
        raise SystemExit(f"No hay archivos (.txt, .md, .pdf) en {DATA_DIR} (búsqueda recursiva).")
    for p in paths:
        # Cada archivo se trocea y se sube con un doc_id igual al nombre base.
        text = read_text(p)
        chunks = split(text)
        upsert(os.path.basename(p), chunks)
    print("✅ Ingesta completada.")


if __name__ == "__main__":
    main()
elif BACKEND == "lmstudio":
    # Este bloque replica la configuración LM Studio cuando se importa el módulo.
    # Se mantiene por compatibilidad con entornos que importan el script sin ejecutarlo.
    # Embeddings vía LM Studio usando httpx (evita rarezas del SDK)
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

    # Puedes forzar el endpoint en .env: EMBEDDING_ENDPOINT=embeddings (o 'embedding' si alguna build lo expone así)
    EMB_ENDPOINT = (os.getenv("EMBEDDING_ENDPOINT") or "embeddings").strip("/")
    EMB_URL = f"{EMB_BASE}/{EMB_ENDPOINT}"

    TIMEOUT = httpx.Timeout(connect=5, read=120, write=30, pool=5)
    HEADERS = {"Authorization": f"Bearer {EMB_KEY}"}
    _embedding_manager = LMStudioManager(base_url=EMB_BASE, api_key=EMB_KEY)
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

    def _post_embeddings(texts):
        _ensure_embedding_ready()
        payload = {"model": EMB_MODEL, "input": texts}
        r = httpx.post(EMB_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 404 and EMB_ENDPOINT == "embeddings":
            # fallback a singular si tu servidor lo usa
            r = httpx.post(f"{EMB_BASE}/embedding", json=payload, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return [d["embedding"] for d in data["data"]]

    print(f"🔎 Embeddings LM Studio → {EMB_URL} | model={EMB_MODEL}")

    EMB_DIM_ENV = os.getenv("EMBEDDING_DIM")
    if EMB_DIM_ENV:
        EMB_DIM = int(EMB_DIM_ENV)
    else:
        # Probe 1 solo para obtener la dimensión
        probe = _post_embeddings(["hola"])
        EMB_DIM = len(probe[0])

    def embed_batch(texts: List[str]) -> List[List[float]]:
        return _post_embeddings(texts)

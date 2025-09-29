from typing import List, Dict, Tuple
import os
from sentence_transformers import CrossEncoder

# Carga lazy (para no penalizar arranque si está desactivado)
_model = None

def _get_model():
    global _model
    if _model is None:
        name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _model = CrossEncoder(name)  # CPU va bien con este modelo pequeño
    return _model

def rerank(query: str, rows: List[Dict], top_k: int) -> List[Dict]:
    """Devuelve rows reordenados por score del cross-encoder, cortados a top_k."""
    if not rows:
        return rows
    model = _get_model()
    pairs = [(query, (r["content"] or "")) for r in rows]
    scores = model.predict(pairs, convert_to_numpy=True)  # mayor = mejor
    # adjunta y ordena
    augmented: List[Tuple[float, Dict]] = [(float(s), r) for s, r in zip(scores, rows)]
    augmented.sort(key=lambda x: x[0], reverse=True)
    return [r for s, r in augmented[:top_k]]

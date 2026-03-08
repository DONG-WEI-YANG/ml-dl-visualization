"""FAISS vector index for semantic search using TF-IDF vectors."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("faiss not available — FAISS vector search disabled")

from .trainer import MODEL_DIR, load_model

# ── Cached FAISS index and metadata ──
_faiss_index = None
_faiss_meta = None
_corpus_data = None
_loaded = False


def _load_index():
    """Lazy-load FAISS index and metadata."""
    global _faiss_index, _faiss_meta, _corpus_data, _loaded
    if _loaded:
        return
    _loaded = True

    if faiss is None:
        logger.info("FAISS not available — skipping index load")
        return

    index_path = MODEL_DIR / "faiss_index.pkl"
    meta_path = MODEL_DIR / "faiss_meta.pkl"

    if not index_path.exists() or not meta_path.exists():
        logger.info("FAISS index not found — run train_models() first")
        return

    try:
        import pickle
        # Load serialized FAISS index (avoids Unicode path issues on Windows)
        with open(index_path, "rb") as f:
            index_bytes = pickle.load(f)
        _faiss_index = faiss.deserialize_index(index_bytes)
        with open(meta_path, "rb") as f:
            _faiss_meta = pickle.load(f)
        _corpus_data = load_model("corpus_tfidf")
        logger.info("FAISS index loaded: %d vectors", _faiss_index.ntotal)
    except Exception as e:
        logger.warning("Failed to load FAISS index: %s", e)
        _faiss_index = None
        _faiss_meta = None


def reload_index():
    """Force reload after retraining."""
    global _loaded
    _loaded = False


def search_similar(query: str, top_k: int = 5) -> list[dict]:
    """Search for similar documents using FAISS inner product (cosine on L2-normalized)."""
    _load_index()

    if _faiss_index is None or _corpus_data is None or _faiss_meta is None:
        return []

    try:
        vectorizer = _corpus_data["vectorizer"]
        query_vec = vectorizer.transform([query]).toarray().astype(np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = _faiss_index.search(query_vec, top_k)

        results = []
        meta_list = _faiss_meta.get("meta", [])
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(meta_list):
                continue
            meta = meta_list[idx]
            results.append({
                "id": meta.get("id"),
                "week": meta.get("week", 0),
                "title": meta.get("title", ""),
                "score": round(float(score), 4),
            })

        return results
    except Exception as e:
        logger.warning("FAISS search failed: %s", e)
        return []


def search_by_id(chunk_ids: list[int], query: str) -> list[tuple[int, float]]:
    """Get FAISS cosine scores for specific chunk IDs against a query."""
    _load_index()

    if _faiss_index is None or _corpus_data is None or _faiss_meta is None:
        return [(cid, 0.0) for cid in chunk_ids]

    try:
        vectorizer = _corpus_data["vectorizer"]
        query_vec = vectorizer.transform([query]).toarray().astype(np.float32)
        faiss.normalize_L2(query_vec)

        # Get all meta IDs for mapping
        meta_list = _faiss_meta.get("meta", [])
        id_to_idx = {}
        for i, meta in enumerate(meta_list):
            id_to_idx[meta.get("id")] = i

        results = []
        for cid in chunk_ids:
            idx = id_to_idx.get(cid)
            if idx is not None and idx < _faiss_index.ntotal:
                # Reconstruct vector and compute similarity
                vec = np.zeros((1, _faiss_index.d), dtype=np.float32)
                _faiss_index.reconstruct(idx, vec[0])
                score = float(np.dot(query_vec[0], vec[0]))
                results.append((cid, score))
            else:
                results.append((cid, 0.0))

        return results
    except Exception as e:
        logger.warning("FAISS by-ID search failed: %s", e)
        return [(cid, 0.0) for cid in chunk_ids]

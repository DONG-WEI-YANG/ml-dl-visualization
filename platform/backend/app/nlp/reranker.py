"""Layer 6: RAG Retrieval + Re-ranking — TF-IDF cosine similarity + FAISS enhanced."""

import logging
from .pipeline import NLPContext
from .trainer import load_model
from app.rag.retriever import retrieve_context
from app.rag.store import search_fts

logger = logging.getLogger(__name__)

_bm25_cls = None


def _get_bm25_cls():
    """Lazy-load BM25Okapi from rank_bm25."""
    global _bm25_cls
    if _bm25_cls is None:
        try:
            from rank_bm25 import BM25Okapi
            _bm25_cls = BM25Okapi
        except ImportError:
            _bm25_cls = False
    return _bm25_cls if _bm25_cls is not False else None


# ── Cached corpus model ──
_corpus_data = None
_corpus_loaded = False


def _get_corpus():
    """Lazy-load the corpus TF-IDF data."""
    global _corpus_data, _corpus_loaded
    if not _corpus_loaded:
        _corpus_data = load_model("corpus_tfidf")
        _corpus_loaded = True
    return _corpus_data


def reload_model():
    """Force reload after retraining."""
    global _corpus_loaded
    _corpus_loaded = False


def _tfidf_rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Re-rank FTS results using TF-IDF cosine similarity."""
    corpus = _get_corpus()
    if corpus is None or not results:
        return results[:top_k]

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = corpus["vectorizer"]
        corpus_matrix = corpus["matrix"]
        corpus_ids = corpus["ids"]

        # Transform query
        query_vec = vectorizer.transform([query])

        # Find corpus indices for our result IDs
        id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
        scored = []
        for r in results:
            idx = id_to_idx.get(r.get("id"))
            if idx is not None:
                sim = float(cosine_similarity(query_vec, corpus_matrix[idx:idx + 1])[0, 0])
                scored.append((r, sim))
            else:
                scored.append((r, 0.0))

        # Sort by cosine similarity
        scored.sort(key=lambda x: -x[1])
        return [r for r, _ in scored[:top_k]]
    except Exception as e:
        logger.warning("TF-IDF rerank failed: %s", e)
        return results[:top_k]


def _bm25_rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Re-rank FTS results using BM25Okapi (better IR ranking than TF-IDF cosine)."""
    BM25Okapi = _get_bm25_cls()
    if BM25Okapi is None or not results:
        return results[:top_k]

    try:
        import jieba
        # Tokenize documents and query with jieba for CJK support
        docs_tokenized = [list(jieba.cut(r.get("content", ""))) for r in results]
        query_tokenized = list(jieba.cut(query))
        bm25 = BM25Okapi(docs_tokenized)
        scores = bm25.get_scores(query_tokenized)
        scored = sorted(zip(results, scores), key=lambda x: -x[1])
        return [r for r, _ in scored[:top_k]]
    except Exception as e:
        logger.warning("BM25 rerank failed: %s", e)
        return results[:top_k]


def _faiss_rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Re-rank FTS results using FAISS cosine similarity."""
    if not results:
        return results[:top_k]

    try:
        from .vector_store import search_by_id
        chunk_ids = [r.get("id") for r in results if r.get("id") is not None]
        if not chunk_ids:
            return results[:top_k]

        id_scores = search_by_id(chunk_ids, query)
        score_map = dict(id_scores)

        scored = [(r, score_map.get(r.get("id"), 0.0)) for r in results]
        scored.sort(key=lambda x: -x[1])
        return [r for r, _ in scored[:top_k]]
    except Exception as e:
        logger.warning("FAISS rerank failed: %s", e)
        return results[:top_k]


def retrieve_and_rerank(ctx: NLPContext) -> NLPContext:
    """Retrieve RAG context with TF-IDF + FAISS cosine similarity re-ranking."""

    # Primary search: use original user message
    context = retrieve_context(ctx.user_message, week=ctx.week, top_k=5)

    # Secondary search: use extracted keywords (often cleaner)
    if len(context) < 200 and ctx.keywords:
        kw_query = " ".join(ctx.keywords[:5])
        extra = retrieve_context(kw_query, week=ctx.week, top_k=3)
        if extra:
            context = context + "\n\n---\n\n" + extra if context else extra

    # Tertiary search: use domain concepts (most precise)
    if len(context) < 200 and ctx.domain_concepts:
        for concept in ctx.domain_concepts[:2]:
            clean_concept = concept.split("(")[0].strip()
            results = search_fts(clean_concept, week=ctx.week, top_k=2)
            if results:
                # Re-ranking cascade: BM25 → FAISS → TF-IDF
                reranked = _bm25_rerank(ctx.user_message, results, top_k=2)
                if all(r == results[i] for i, r in enumerate(reranked)):
                    reranked = _faiss_rerank(ctx.user_message, results, top_k=2)
                if all(r == results[i] for i, r in enumerate(reranked)):
                    reranked = _tfidf_rerank(ctx.user_message, results, top_k=2)

                extra_parts = [
                    f"[來源：第{r['week']}週 {r['file_type']} — {r['title']}]\n{r['content']}"
                    for r in reranked
                ]
                extra = "\n\n---\n\n".join(extra_parts)
                context = context + "\n\n---\n\n" + extra if context else extra
                break

    # FAISS secondary retrieval channel: find semantically similar chunks
    if len(context) < 200:
        try:
            from .vector_store import search_similar
            faiss_results = search_similar(ctx.user_message, top_k=3)
            if faiss_results:
                for fr in faiss_results:
                    if fr.get("score", 0) > 0.3:
                        fts_results = search_fts(fr.get("title", ""), week=fr.get("week"), top_k=1)
                        if fts_results:
                            extra_parts = [
                                f"[來源：第{r['week']}週 {r['file_type']} — {r['title']}]\n{r['content']}"
                                for r in fts_results
                            ]
                            extra = "\n\n---\n\n".join(extra_parts)
                            context = context + "\n\n---\n\n" + extra if context else extra
                            break
        except Exception:
            pass

    ctx.rag_context = context

    # Track sources
    if context:
        import re
        sources = re.findall(r"\[來源：([^\]]+)\]", context)
        ctx.rag_sources = sources

    return ctx

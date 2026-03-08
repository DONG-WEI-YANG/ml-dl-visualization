"""Group F: Retrieval Enhancement Layers (L33-36).

L33 QueryExpander — jieba keyword extraction + synonym map + concept map
L34 RAGRetriever — existing FTS5 (delegates to rag.retriever)
L35 SemanticReranker — sentence-transformers cosine similarity
L36 CrossWeekLinker — concept-based cross-week linking + FAISS similarity
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")
        except Exception:
            _sentence_model = False
    return _sentence_model if _sentence_model is not False else None


# ── Simple synonym map for query expansion ──

SYNONYMS = {
    "準確率": ["accuracy", "精確度"],
    "損失": ["loss", "損失函數", "cost"],
    "梯度下降": ["gradient descent", "GD"],
    "學習率": ["learning rate", "lr"],
    "過擬合": ["overfitting", "過度擬合"],
    "欠擬合": ["underfitting", "擬合不足"],
    "隨機森林": ["random forest", "RF"],
    "決策樹": ["decision tree"],
    "激活函數": ["activation function"],
    "反向傳播": ["backpropagation", "back propagation"],
    "注意力": ["attention", "self-attention"],
    "卷積": ["convolution", "conv"],
}

# ── Week concept links for cross-week navigation ──

CONCEPT_WEEK_MAP = {
    "梯度下降": [4, 11], "損失函數": [4, 14], "學習率": [4, 10, 14],
    "過擬合": [3, 7, 10, 14], "欠擬合": [3, 10],
    "決策邊界": [5, 6], "SVM": [5, 6], "特徵工程": [8, 9],
    "正則化": [11, 14], "CNN": [12], "RNN": [13], "Transformer": [13, 17],
    "SHAP": [8], "公平性": [15], "MLOps": [16], "嵌入": [17],
    "交叉驗證": [3, 10], "超參數": [10],
}


# ── L33: Query Expander (enhanced with jieba keyword extraction) ──

def query_expander(ctx: NLPContext) -> NLPContext:
    """L33: Expand query using jieba keywords + synonyms + domain concept map."""
    expanded_parts = [ctx.user_message]

    # Use jieba for keyword extraction to find expansion targets
    jieba_keywords = []
    try:
        import jieba.analyse
        jieba_keywords = jieba.analyse.extract_tags(ctx.user_message, topK=5)
    except ImportError:
        pass

    # Combine with pipeline-extracted keywords
    all_keywords = list(set(jieba_keywords + ctx.keywords[:5]))

    # Add synonyms for detected keywords
    for kw in all_keywords:
        kw_lower = kw.lower()
        for trigger, syns in SYNONYMS.items():
            if trigger in kw_lower or kw_lower in trigger.lower():
                expanded_parts.extend(syns[:2])

    # Add related terms from domain concept map (from topic.py)
    try:
        from .topic import CONCEPT_MAP as TOPIC_CONCEPT_MAP
        for kw in all_keywords:
            kw_lower = kw.lower()
            for trigger, concept in TOPIC_CONCEPT_MAP.items():
                if trigger in kw_lower or kw_lower in trigger:
                    # Add the bilingual concept as expansion
                    expanded_parts.append(concept)
                    break
    except ImportError:
        pass

    ctx.expanded_query = " ".join(expanded_parts)
    return ctx


# ── L34: RAG Retriever ──

def rag_retriever(ctx: NLPContext) -> NLPContext:
    """L34: Retrieve context using FTS5 (delegates to existing retriever)."""
    from app.rag.retriever import retrieve_context

    # Use expanded query if available
    query = ctx.expanded_query or ctx.user_message
    context = retrieve_context(query, week=ctx.week, top_k=5)

    # If sparse, try with just keywords
    if len(context) < 200 and ctx.keywords:
        kw_query = " ".join(ctx.keywords[:5])
        extra = retrieve_context(kw_query, week=ctx.week, top_k=3)
        if extra:
            context = context + "\n\n---\n\n" + extra if context else extra

    ctx.rag_context = context

    # Track sources
    if context:
        sources = re.findall(r"\[來源：([^\]]+)\]", context)
        ctx.rag_sources = sources

    return ctx


# ── L35: Semantic Reranker ──

def semantic_reranker(ctx: NLPContext) -> NLPContext:
    """L35: Re-rank RAG results by semantic similarity (if sentence-transformers available)."""
    model = _get_sentence_model()
    if model is None or not ctx.rag_context:
        return ctx

    try:
        # Split context into chunks
        chunks = ctx.rag_context.split("\n\n---\n\n")
        if len(chunks) <= 1:
            return ctx

        # Encode query and chunks
        texts = [ctx.user_message] + chunks
        embeddings = model.encode(texts)

        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = embeddings[0:1]
        chunk_embs = embeddings[1:]
        sims = cosine_similarity(query_emb, chunk_embs)[0]

        # Re-order by similarity
        ranked = sorted(zip(chunks, sims), key=lambda x: -x[1])
        ctx.rag_context = "\n\n---\n\n".join([c for c, _ in ranked])
        ctx.reranked_results = [{"chunk": c[:100], "score": round(float(s), 3)} for c, s in ranked]
    except Exception as e:
        logger.warning("Semantic reranking failed: %s", e)

    return ctx


# ── L36: Cross-Week Linker (enhanced with FAISS similarity) ──

def cross_week_linker(ctx: NLPContext) -> NLPContext:
    """L36: Find related content in other weeks using concept map + FAISS similarity."""
    links = []
    current_week = ctx.week

    # Concept-based linking (original approach)
    for concept in ctx.domain_concepts:
        # Extract the Chinese part of the concept
        clean = concept.split("(")[0].strip()
        related_weeks = CONCEPT_WEEK_MAP.get(clean, [])
        for w in related_weeks:
            if w != current_week:
                links.append({"week": w, "concept": concept, "relation": "相關概念"})

    # FAISS-based cross-week similarity search
    try:
        from .vector_store import search_similar
        results = search_similar(ctx.user_message, top_k=5)
        for result in results:
            w = result.get("week", 0)
            if w != current_week and w > 0:
                title = result.get("title", "")
                score = result.get("score", 0)
                if score > 0.3:  # Only include reasonably similar results
                    links.append({
                        "week": w,
                        "concept": title,
                        "relation": f"語義相似 (score={score:.2f})",
                    })
    except Exception:
        pass  # FAISS not available or not trained yet

    # Deduplicate by week
    seen_weeks = set()
    unique_links = []
    for link in links:
        if link["week"] not in seen_weeks:
            seen_weeks.add(link["week"])
            unique_links.append(link)

    ctx.cross_week_links = unique_links[:5]
    return ctx


# ── Public aliases (used by __init__.py FULL_PIPELINE) ──

expand_query = query_expander
link_cross_week = cross_week_linker

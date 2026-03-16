"""Group G: Response Generation Layers (L37-42).

L37 ResponseAssembler — enhanced adaptive template
L38 ComplexityAdjuster — adjust based on student level
L39 CitationInjector — add week/source references
L40 FollowUpGenerator — generate guiding questions
L41 EncouragementGenerator — emotion-aware encouragement
L42 ResponseCompletenessChecker — check response quality
"""

import jieba
import jieba.analyse
import jieba.posseg as posseg
import textstat
import numpy as np
from snownlp import SnowNLP
from rapidfuzz import fuzz, process as rfprocess
from wordfreq import word_frequency

from .pipeline import NLPContext
from .response import (
    INTENT_TEMPLATES, EMOTION_PREFIX, LEVEL_GUIDANCE,
    HINT_LADDER, INTENT_GUIDANCE, NO_CONTEXT, HOMEWORK_GUARD,
    _concept_note,
)


# ── L37: Response Assembler (enhanced) ──

def response_assembler(ctx: NLPContext) -> NLPContext:
    """L37: Assemble response using all NLP layer outputs."""
    topic = "\u3001".join(ctx.keywords[:3]) if ctx.keywords else ctx.user_message[:20]

    # No context found
    if not ctx.rag_context:
        ctx.response = NO_CONTEXT.format(
            keywords="\u3001".join(ctx.keywords) if ctx.keywords else ctx.user_message[:30]
        )
        return ctx

    # Homework mode
    if ctx.is_homework:
        ctx.response = HOMEWORK_GUARD.format(context=ctx.rag_context)
        return ctx

    parts = []

    # 1. Emotion prefix
    prefix = EMOTION_PREFIX.get(ctx.emotion, "")
    if prefix:
        parts.append(prefix)

    # 2. Main content
    template = INTENT_TEMPLATES.get(ctx.intent, INTENT_TEMPLATES["general"])
    parts.append(template.format(topic=topic, context=ctx.rag_context))

    # 3. Misconception warnings
    if ctx.misconceptions:
        parts.append("\n\n---\n" + "\n".join(ctx.misconceptions[:2]))

    # 4. Domain concept note
    concept_note = _concept_note(ctx)
    if concept_note:
        parts.append(concept_note)

    # 5. Separator
    parts.append("\n---")

    # 6. Level guidance
    level_guides = LEVEL_GUIDANCE.get(ctx.student_level, LEVEL_GUIDANCE["beginner"])
    guide = level_guides.get(ctx.intent, level_guides.get("general", ""))
    if guide:
        parts.append(guide)

    # 7. Learning style hint
    if ctx.learning_style == "visual":
        parts.append("\n\ud83c\udfa8 **\u5efa\u8b70\uff1a** \u4f60\u504f\u597d\u8996\u89ba\u5316\u5b78\u7fd2\uff0c\u8a66\u8a66\u5e73\u53f0\u4e0a\u7684\u4e92\u52d5\u8996\u89ba\u5316\u5de5\u5177\uff01")
    elif ctx.learning_style == "practical":
        parts.append("\n\ud83d\udcbb **\u5efa\u8b70\uff1a** \u4f60\u504f\u597d\u52d5\u624b\u5be6\u4f5c\uff0c\u5efa\u8b70\u5148\u8dd1\u4e00\u904d Notebook \u518d\u770b\u8b1b\u7fa9\u3002")

    # 8. Cross-week links
    if ctx.cross_week_links:
        links = ctx.cross_week_links[:3]
        link_text = "\u3001".join([f"\u7b2c{l['week']}\u9031\uff08{l['concept']}\uff09" for l in links])
        parts.append(f"\n\n\ud83d\udd17 **\u76f8\u95dc\u9031\u6b21\uff1a** {link_text}")

    ctx.response = "".join(parts)
    return ctx


# ── L38: Complexity Adjuster ──

def complexity_adjuster(ctx: NLPContext) -> NLPContext:
    """L38: Adjust response complexity based on student level."""
    if not ctx.response:
        return ctx

    # --- NLP: Measure actual complexity with textstat ---
    try:
        reading_ease = textstat.flesch_reading_ease(ctx.response)
        text_standard = textstat.text_standard(ctx.response, float_output=True)
    except Exception:
        reading_ease = 50.0
        text_standard = 6.0

    # --- NLP: Count technical vs common terms using jieba + wordfreq ---
    tokens = jieba.lcut(ctx.response)
    technical_count = 0
    common_count = 0
    for token in tokens:
        if len(token) < 2:
            continue
        freq = word_frequency(token, "zh")
        if freq < 1e-6:
            technical_count += 1  # Rare word = likely technical
        else:
            common_count += 1
    total_meaningful = technical_count + common_count
    tech_ratio = technical_count / total_meaningful if total_meaningful > 0 else 0.0

    # --- Determine complexity from actual measurement ---
    # reading_ease: higher = easier; text_standard: higher = harder
    if reading_ease > 60 and tech_ratio < 0.3:
        measured_complexity = "simple"
    elif reading_ease < 30 or tech_ratio > 0.5:
        measured_complexity = "advanced"
    else:
        measured_complexity = "moderate"

    # --- Original logic (kept as baseline) ---
    if ctx.student_level == "beginner" and ctx.question_complexity == "simple":
        ctx.response_complexity = "simple"
        if len(ctx.response) > 500:
            ctx.response += "\n\n\ud83d\udcdd **\u7c21\u55ae\u4f86\u8aaa\uff1a** \u4e0a\u9762\u7684\u5167\u5bb9\u6bd4\u8f03\u591a\uff0c\u5efa\u8b70\u5148\u8b80\u6a19\u793a\u70ba\u300c\u521d\u5b78\u8005\u5efa\u8b70\u300d\u7684\u90e8\u5206\u3002"
    elif ctx.student_level == "advanced":
        ctx.response_complexity = "advanced"
    else:
        ctx.response_complexity = "moderate"

    # --- NLP: Override if measured complexity mismatches student level ---
    if ctx.student_level == "beginner" and measured_complexity == "advanced":
        ctx.response_complexity = "moderate"
        ctx.response += "\n\n\ud83d\udcdd **\u8907\u96dc\u5ea6\u63d0\u793a\uff1a** \u9019\u500b\u56de\u7b54\u5305\u542b\u8f03\u591a\u5c08\u696d\u8853\u8a9e\uff0c\u5efa\u8b70\u5148\u5c08\u6ce8\u7406\u89e3\u6a19\u793a\u70ba\u300c\u521d\u5b78\u8005\u5efa\u8b70\u300d\u7684\u90e8\u5206\u3002"
    elif ctx.student_level == "advanced" and measured_complexity == "simple":
        ctx.response_complexity = "moderate"

    return ctx


# ── L39: Citation Injector ──

# Known curriculum title patterns for fuzzy matching
_CURRICULUM_TITLES = [
    "機器學習基礎", "深度學習導論", "神經網路", "梯度下降",
    "反向傳播", "卷積神經網路", "循環神經網路", "注意力機制",
    "Transformer", "GAN", "強化學習", "自然語言處理",
    "資料預處理", "特徵工程", "模型評估", "超參數調整",
    "正規化", "最佳化", "損失函數", "啟動函數",
]


def citation_injector(ctx: NLPContext) -> NLPContext:
    """L39: Add explicit references to curriculum materials."""
    # --- Original: use rag_sources ---
    unique_sources = list(dict.fromkeys(ctx.rag_sources))[:3] if ctx.rag_sources else []

    # --- NLP: Extract proper nouns from RAG context via jieba.posseg ---
    nlp_sources = []
    if ctx.rag_context:
        try:
            words = posseg.lcut(ctx.rag_context)
            # nr=person, ns=place, nt=org, nz=other proper noun
            proper_nouns = [w.word for w in words if w.flag in ("nr", "ns", "nt", "nz") and len(w.word) >= 2]
            # Deduplicate while preserving order
            seen = set()
            for pn in proper_nouns:
                if pn not in seen:
                    seen.add(pn)
                    nlp_sources.append(pn)
        except Exception:
            pass

    # --- NLP: Fuzzy match extracted sources against curriculum titles ---
    matched_titles = []
    all_candidates = unique_sources + nlp_sources
    for candidate in all_candidates:
        try:
            matches = rfprocess.extract(candidate, _CURRICULUM_TITLES, scorer=fuzz.partial_ratio, limit=1)
            if matches and matches[0][1] >= 70:
                matched_titles.append(f"{candidate}（相關：{matches[0][0]}）")
            else:
                matched_titles.append(candidate)
        except Exception:
            matched_titles.append(candidate)

    # Deduplicate and limit
    final_sources = list(dict.fromkeys(matched_titles))[:5]
    if not final_sources:
        final_sources = unique_sources

    ctx.citations = final_sources if final_sources else unique_sources

    if ctx.citations:
        citation_text = "\n\n[Ref] **參考來源：**\n" + "\n".join([f"- {s}" for s in ctx.citations])
        ctx.response += citation_text

    return ctx


# ── L40: Follow-Up Question Generator ──

FOLLOWUP_TEMPLATES = {
    "definition": ["\u4f60\u80fd\u7528\u81ea\u5df1\u7684\u8a71\u91cd\u65b0\u89e3\u91cb\u9019\u500b\u6982\u5ff5\u55ce\uff1f", "\u4f60\u89ba\u5f97\u9019\u548c{related}\u6709\u4ec0\u9ebc\u95dc\u4fc2\uff1f"],
    "how": ["\u4f60\u6709\u5617\u8a66\u81ea\u5df1\u5be6\u4f5c\u55ce\uff1f\u54ea\u4e00\u6b65\u5361\u4f4f\u4e86\uff1f", "\u4f60\u89ba\u5f97\u6bcf\u4e00\u6b65\u7684\u76ee\u7684\u662f\u4ec0\u9ebc\uff1f"],
    "why": ["\u5982\u679c\u4e0d\u9019\u6a23\u505a\uff0c\u4f60\u89ba\u5f97\u6703\u767c\u751f\u4ec0\u9ebc\uff1f", "\u4f60\u80fd\u60f3\u5230\u4e00\u500b\u53cd\u4f8b\u55ce\uff1f"],
    "debug": ["\u4f60\u80fd\u628a\u5b8c\u6574\u7684\u932f\u8aa4\u8a0a\u606f\u8cbc\u51fa\u4f86\u55ce\uff1f", "\u4f60\u662f\u5728\u54ea\u4e00\u884c\u51fa\u932f\u7684\uff1f"],
    "compare": ["\u5728\u4f60\u76ee\u524d\u7684\u5c08\u6848\u4e2d\uff0c\u4f60\u6703\u9078\u64c7\u54ea\u4e00\u500b\uff1f\u70ba\u4ec0\u9ebc\uff1f"],
    "code": ["\u4f60\u80fd\u89e3\u91cb\u9019\u6bb5\u7a0b\u5f0f\u78bc\u6bcf\u4e00\u884c\u5728\u505a\u4ec0\u9ebc\u55ce\uff1f"],
    "general": ["\u95dc\u65bc\u9019\u500b\u4e3b\u984c\uff0c\u4f60\u6700\u60f3\u4e86\u89e3\u7684\u662f\u4ec0\u9ebc\uff1f"],
}


# Domain concept map for follow-up question generation
_DOMAIN_CONCEPT_MAP = [
    "梯度下降", "學習率", "損失函數", "反向傳播", "過擬合",
    "正規化", "批次大小", "激活函數", "卷積", "池化",
    "注意力機制", "Transformer", "嵌入", "特徵提取",
    "資料增強", "交叉驗證", "混淆矩陣", "精確率", "召回率",
    "F1 分數", "偏差", "變異數", "Dropout", "批次正規化",
    "遷移學習", "微調", "生成對抗網路", "自編碼器",
]


def follow_up_generator(ctx: NLPContext) -> NLPContext:
    """L40: Generate follow-up questions to guide student thinking."""
    templates = FOLLOWUP_TEMPLATES.get(ctx.intent, FOLLOWUP_TEMPLATES["general"])

    # Pick template based on hint level
    idx = min(ctx.hint_level - 1, len(templates) - 1)
    question = templates[idx]

    # --- NLP: Extract keywords from user message ---
    user_keywords = []
    try:
        user_keywords = jieba.analyse.extract_tags(ctx.user_message, topK=5)
    except Exception:
        pass

    # --- NLP: Extract keywords via SnowNLP ---
    snow_keywords = []
    try:
        snow_keywords = SnowNLP(ctx.user_message).keywords(3)
    except Exception:
        pass

    # Merge NLP keywords
    all_keywords = list(dict.fromkeys(user_keywords + snow_keywords))

    # --- NLP: Find related concepts via rapidfuzz fuzzy matching ---
    related_concepts = []
    if all_keywords:
        for kw in all_keywords[:3]:
            try:
                matches = rfprocess.extract(kw, _DOMAIN_CONCEPT_MAP, scorer=fuzz.ratio, limit=2)
                for match_text, score, _ in matches:
                    if score >= 50 and match_text not in related_concepts:
                        related_concepts.append(match_text)
            except Exception:
                pass

    # Fill in related concept with NLP-found concept (or domain_concepts fallback)
    related = (
        related_concepts[0] if related_concepts
        else ctx.domain_concepts[0] if ctx.domain_concepts
        else "其他相關概念"
    )
    if "{related}" in question:
        question = question.format(related=related)
    else:
        question = question.replace("{related}", related)

    ctx.follow_up_questions = [question]

    # Add NLP-generated concept-specific follow-up if related concepts found
    if related_concepts and len(related_concepts) > 1:
        extra_concept = related_concepts[1]
        extra_q = f"你知道「{extra_concept}」和這個主題的關聯嗎？"
        ctx.follow_up_questions.append(extra_q)

    ctx.response += f"\n\n\u2753 **\u601d\u8003\u984c\uff1a** {ctx.follow_up_questions[0]}"

    return ctx


# ── L41: Encouragement Generator ──

ENCOURAGEMENTS = {
    "frustrated": [
        "\u5225\u64d4\u5fc3\uff0c\u9019\u500b\u6982\u5ff5\u78ba\u5be6\u4e0d\u5bb9\u6613\uff0c\u5f88\u591a\u540c\u5b78\u4e5f\u82b1\u4e86\u4e0d\u5c11\u6642\u9593\u624d\u7406\u89e3\u3002",
        "\u4f60\u5df2\u7d93\u5f88\u52aa\u529b\u4e86\uff01\u9047\u5230\u56f0\u96e3\u662f\u9032\u6b65\u7684\u5fc5\u7d93\u4e4b\u8def\u3002",
        "\u4e00\u6b65\u4e00\u6b65\u4f86\uff0c\u4e0d\u9700\u8981\u4e00\u6b21\u5168\u90e8\u641e\u61c2\u3002",
    ],
    "confused": [
        "\u6df7\u6dc6\u662f\u6b63\u5e38\u7684\uff01\u5f88\u591a\u6982\u5ff5\u5728\u521d\u5b78\u6642\u78ba\u5be6\u5bb9\u6613\u641e\u6df7\u3002",
        "\u82b1\u6642\u9593\u91d0\u6e05\u9019\u4e9b\u6982\u5ff5\u662f\u503c\u5f97\u7684\uff0c\u4e4b\u5f8c\u6703\u8d8a\u4f86\u8d8a\u6e05\u695a\u3002",
    ],
    "curious": [
        "\u4f60\u7684\u597d\u5947\u5fc3\u5f88\u68d2\uff01\u4fdd\u6301\u9019\u7a2e\u6c42\u77e5\u6170\uff01",
        "\u554f\u51fa\u597d\u554f\u984c\u672c\u8eab\u5c31\u662f\u4e00\u7a2e\u80fd\u529b\uff01",
    ],
    "confident": [
        "\u5f88\u9ad8\u8208\u4f60\u6709\u81ea\u4fe1\uff01\u8b93\u6211\u5011\u4f86\u9a57\u8b49\u4f60\u7684\u7406\u89e3\u3002",
    ],
}


def encouragement_generator(ctx: NLPContext) -> NLPContext:
    """L41: Add emotion-aware encouragement."""
    import random

    # --- NLP: Fine-grained sentiment via SnowNLP ---
    try:
        nlp_sentiment = SnowNLP(ctx.user_message).sentiments
    except Exception:
        nlp_sentiment = 0.5

    # --- NLP: Compute weighted encouragement intensity via numpy ---
    # Factors: frustration_level (0-5), sentiment (0-1), confidence (0-1), turn_count
    factors = np.array([
        ctx.frustration_level / 5.0,        # Higher frustration = more encouragement
        1.0 - nlp_sentiment,                # Lower sentiment = more encouragement
        1.0 - ctx.confidence_level,         # Lower confidence = more encouragement
        min(ctx.turn_count / 5.0, 1.0),     # More turns = more encouragement
    ])
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    intensity = float(np.dot(factors, weights))  # 0.0 to 1.0

    # Select encouragement based on intensity, not just emotion category
    if intensity > 0.6:
        # High intensity: use frustrated encouragements
        msgs = ENCOURAGEMENTS.get("frustrated", [])
    elif intensity > 0.4:
        # Medium intensity: use confused encouragements
        msgs = ENCOURAGEMENTS.get("confused", ENCOURAGEMENTS.get(ctx.emotion, []))
    else:
        # Low intensity: use emotion-based or curious
        msgs = ENCOURAGEMENTS.get(ctx.emotion, [])

    if msgs:
        msg = random.choice(msgs)
        ctx.encouragement = msg
        # Show encouragement if intensity is meaningful or emotion calls for it
        if intensity > 0.35 or ctx.emotion in ("frustrated", "confused"):
            ctx.response += f"\n\n\ud83d\udcaa {msg}"

    return ctx


# ── L42: Response Completeness Checker ──

def response_completeness_checker(ctx: NLPContext) -> NLPContext:
    """L42: Check if the response addresses all detected sub-questions."""
    # --- NLP: Measure response substance with textstat ---
    try:
        word_count = textstat.lexicon_count(ctx.response, removepunct=True)
        sentence_count = textstat.sentence_count(ctx.response)
    except Exception:
        word_count = len(ctx.response)
        sentence_count = max(1, ctx.response.count("。") + ctx.response.count("."))

    # --- NLP: Keyword coverage via jieba + rapidfuzz ---
    user_keywords = []
    response_keywords = []
    try:
        user_keywords = jieba.analyse.extract_tags(ctx.user_message, topK=8)
        response_keywords = jieba.analyse.extract_tags(ctx.response, topK=15)
    except Exception:
        pass

    coverage_score = 0.0
    if user_keywords and response_keywords:
        matched = 0
        for uk in user_keywords:
            # Check if any response keyword fuzzy-matches the user keyword
            try:
                best = rfprocess.extractOne(uk, response_keywords, scorer=fuzz.ratio)
                if best and best[1] >= 60:
                    matched += 1
            except Exception:
                if uk in " ".join(response_keywords):
                    matched += 1
        coverage_score = matched / len(user_keywords) if user_keywords else 0.0

    # --- NLP: Multi-dimensional completeness scoring with numpy ---
    # Dimensions: coverage, length, complexity, follow-up, citations
    dims = np.array([
        coverage_score,                                          # keyword coverage
        min(word_count / 200.0, 1.0),                           # response length adequacy
        1.0 if ctx.rag_context else 0.0,                        # has RAG context
        1.0 if ctx.follow_up_questions else 0.0,                # has follow-up
        min(len(ctx.citations) / 2.0, 1.0) if ctx.citations else 0.0,  # has citations
    ])
    dim_weights = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
    nlp_score = float(np.dot(dims, dim_weights))

    # --- Original boolean checks (kept as baseline) ---
    base_score = 0.5
    missing = []

    if ctx.rag_context:
        base_score += 0.2

    if ctx.intent != "general" and ctx.intent in ctx.response.lower():
        base_score += 0.1

    if ctx.follow_up_questions:
        base_score += 0.1

    if ctx.knowledge_gaps and not any(gap in ctx.response for gap in ctx.knowledge_gaps):
        missing.append("\u5148\u5099\u77e5\u8b58\u63d0\u793a")

    if ctx.misconceptions and not any("\u8ff7\u601d" in ctx.response for _ in ctx.misconceptions):
        missing.append("\u8ff7\u601d\u6982\u5ff5\u63d0\u9192")

    # --- Combine original + NLP scores ---
    combined_score = 0.5 * min(base_score, 1.0) + 0.5 * nlp_score
    ctx.completeness_score = min(combined_score, 1.0)
    ctx.completeness_missing = missing

    return ctx


# ── Public aliases (used by __init__.py FULL_PIPELINE) ──

adjust_complexity = complexity_adjuster
inject_citations = citation_injector
generate_follow_up = follow_up_generator
generate_encouragement = encouragement_generator
check_completeness = response_completeness_checker

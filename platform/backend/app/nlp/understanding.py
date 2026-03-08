"""Group B: Student Understanding Layers (L8, L10-14).

L7 IntentClassifier — existing intent.py
L8 SubIntentDetector — sklearn sub-categories
L9 EmotionClassifier — existing emotion.py
L10 SentimentScorer — snownlp
L11 FrustrationEscalator — multi-turn escalation
L12 ConfidenceEstimator — snownlp + pattern
L13 UrgencyDetector — pattern + time NER
L14 PolitenessDetector — snownlp + pattern
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_snownlp = None


def _get_snownlp():
    global _snownlp
    if _snownlp is None:
        from snownlp import SnowNLP
        _snownlp = SnowNLP
    return _snownlp


# ── L8: Sub-Intent Detector ──

SUB_INTENTS = {
    "definition": ["basic_definition", "formal_definition", "intuitive_explanation"],
    "how": ["step_by_step", "code_implementation", "tool_usage"],
    "debug": ["error_message", "wrong_output", "environment_issue"],
    "compare": ["pros_cons", "when_to_use", "performance_comparison"],
    "code": ["syntax_help", "library_usage", "full_example"],
}


def sub_intent_detector(ctx: NLPContext) -> NLPContext:
    """L8: Detect sub-intent within the main intent category."""
    text_lower = ctx.user_message.lower()
    subs = SUB_INTENTS.get(ctx.intent, [])
    if not subs:
        ctx.sub_intent = ""
        ctx.sub_intent_confidence = 0.0
        return ctx

    # Simple heuristic matching
    if ctx.intent == "definition":
        if any(w in text_lower for w in ["直覺", "簡單", "白話", "intuition", "simple"]):
            ctx.sub_intent = "intuitive_explanation"
        elif any(w in text_lower for w in ["正式", "數學", "formal", "rigorous"]):
            ctx.sub_intent = "formal_definition"
        else:
            ctx.sub_intent = "basic_definition"
    elif ctx.intent == "how":
        if any(w in text_lower for w in ["程式", "code", "python", "import"]):
            ctx.sub_intent = "code_implementation"
        elif any(w in text_lower for w in ["步驟", "step", "流程"]):
            ctx.sub_intent = "step_by_step"
        else:
            ctx.sub_intent = "tool_usage"
    elif ctx.intent == "debug":
        if any(w in text_lower for w in ["error", "錯誤", "traceback", "exception"]):
            ctx.sub_intent = "error_message"
        elif any(w in text_lower for w in ["裝", "install", "版本", "version"]):
            ctx.sub_intent = "environment_issue"
        else:
            ctx.sub_intent = "wrong_output"
    elif ctx.intent == "compare":
        if any(w in text_lower for w in ["優缺", "pros", "cons", "好壞"]):
            ctx.sub_intent = "pros_cons"
        elif any(w in text_lower for w in ["何時", "when", "場景", "scenario"]):
            ctx.sub_intent = "when_to_use"
        else:
            ctx.sub_intent = "performance_comparison"
    elif ctx.intent == "code":
        if any(w in text_lower for w in ["語法", "syntax", "怎麼寫"]):
            ctx.sub_intent = "syntax_help"
        elif any(w in text_lower for w in ["套件", "library", "模組", "import"]):
            ctx.sub_intent = "library_usage"
        else:
            ctx.sub_intent = "full_example"
    else:
        ctx.sub_intent = subs[0] if subs else ""

    ctx.sub_intent_confidence = 0.7
    return ctx


# ── L10: Sentiment Scorer ──

def sentiment_scorer(ctx: NLPContext) -> NLPContext:
    """L10: Continuous sentiment score (0=negative, 1=positive) using SnowNLP."""
    SnowNLP = _get_snownlp()
    try:
        s = SnowNLP(ctx.user_message)
        ctx.sentiment_score = round(s.sentiments, 3)
    except Exception:
        ctx.sentiment_score = 0.5
    return ctx


# ── L11: Frustration Escalator ──

def frustration_escalator(ctx: NLPContext) -> NLPContext:
    """L11: Track frustration level across conversation turns (0-5)."""
    level = 0

    # Base from emotion
    if ctx.emotion == "frustrated":
        level = 3
    elif ctx.emotion == "confused":
        level = 1

    # Escalate based on sentiment
    if ctx.sentiment_score < 0.2:
        level += 1
    elif ctx.sentiment_score < 0.1:
        level += 2

    # Multi-turn escalation
    frustrated_turns = sum(
        1 for m in ctx.conversation_history
        if m.get("role") == "user" and any(
            w in m.get("content", "").lower()
            for w in ["不懂", "不行", "失敗", "stuck", "frustrated"]
        )
    )
    level += min(frustrated_turns, 2)

    ctx.frustration_level = min(level, 5)
    return ctx


# ── L12: Confidence Estimator ──

CONFIDENCE_HIGH = [
    r"我覺得", r"我認為", r"應該是", r"我確定", r"我知道",
    r"i think", r"i believe", r"i'm sure", r"definitely",
]
CONFIDENCE_LOW = [
    r"不確定", r"可能", r"也許", r"不知道", r"大概",
    r"not sure", r"maybe", r"perhaps", r"i guess",
]


def confidence_estimator(ctx: NLPContext) -> NLPContext:
    """L12: Estimate student confidence level (0-1)."""
    text_lower = ctx.user_message.lower()

    high = sum(1 for p in CONFIDENCE_HIGH if re.search(p, text_lower))
    low = sum(1 for p in CONFIDENCE_LOW if re.search(p, text_lower))

    # Combine with sentiment
    base = ctx.sentiment_score * 0.3
    if high > low:
        ctx.confidence_level = min(base + 0.4 + high * 0.1, 1.0)
    elif low > high:
        ctx.confidence_level = max(base + 0.1 - low * 0.1, 0.0)
    else:
        ctx.confidence_level = 0.5

    return ctx


# ── L13: Urgency Detector (enhanced from existing) ──

URGENCY_HIGH = [
    r"急", r"趕", r"明天", r"今天", r"馬上", r"快", r"來不及", r"deadline",
    r"due", r"urgent", r"asap", r"tonight", r"tomorrow", r"趕快", r"拜託",
]
URGENCY_LOW = [
    r"順便", r"閒聊", r"好奇而已", r"隨便問", r"有空",
    r"just wondering", r"no rush", r"by the way", r"whenever",
]


def urgency_detector(ctx: NLPContext) -> NLPContext:
    """L13: Detect urgency level."""
    text_lower = ctx.user_message.lower()
    if any(re.search(p, text_lower) for p in URGENCY_HIGH):
        ctx.urgency = "high"
    elif any(re.search(p, text_lower) for p in URGENCY_LOW):
        ctx.urgency = "low"
    else:
        ctx.urgency = "normal"
    return ctx


# ── L14: Politeness Detector ──

POLITE_SIGNALS = [
    r"請", r"麻煩", r"謝謝", r"感謝", r"不好意思", r"打擾",
    r"please", r"thank", r"sorry", r"appreciate", r"kindly",
]


def politeness_detector(ctx: NLPContext) -> NLPContext:
    """L14: Detect politeness level (0-1) using SnowNLP + patterns."""
    text_lower = ctx.user_message.lower()

    polite_count = sum(1 for p in POLITE_SIGNALS if re.search(p, text_lower))
    pattern_score = min(polite_count * 0.15, 0.5)

    # SnowNLP sentiment as proxy (positive sentiment correlates with politeness)
    sentiment_component = ctx.sentiment_score * 0.3

    ctx.politeness_score = min(0.3 + pattern_score + sentiment_component, 1.0)
    return ctx


# ── Public aliases (used by __init__.py FULL_PIPELINE) ──

detect_sub_intent = sub_intent_detector
score_sentiment = sentiment_scorer
escalate_frustration = frustration_escalator
estimate_confidence = confidence_estimator
detect_politeness = politeness_detector

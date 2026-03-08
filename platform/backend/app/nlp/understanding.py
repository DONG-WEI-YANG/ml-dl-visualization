"""Group B: Student Understanding Layers (L8, L10-14).

L7 IntentClassifier — existing intent.py
L8 SubIntentDetector — sklearn sub-categories
L9 EmotionClassifier — existing emotion.py
L10 SentimentScorer — snownlp
L11 FrustrationEscalator — multi-turn escalation with snownlp trend
L12 ConfidenceEstimator — ML model + regex fallback
L13 UrgencyDetector — pattern + time NER
L14 PolitenessDetector — ML model + snownlp + pattern fallback
"""

import re
import logging
from .pipeline import NLPContext
from .trainer import load_model

logger = logging.getLogger(__name__)

_snownlp = None

# ── Cached ML models ──
_sub_intent_model = None
_sub_intent_loaded = False
_confidence_model = None
_confidence_loaded = False
_politeness_model = None
_politeness_loaded = False


def _get_snownlp():
    global _snownlp
    if _snownlp is None:
        try:
            from snownlp import SnowNLP
            _snownlp = SnowNLP
        except ImportError:
            logger.warning("snownlp not available")
            _snownlp = False
    return _snownlp if _snownlp is not False else None


def _get_sub_intent_model():
    global _sub_intent_model, _sub_intent_loaded
    if not _sub_intent_loaded:
        _sub_intent_model = load_model("sub_intent_model")
        _sub_intent_loaded = True
        if _sub_intent_model:
            logger.info("Sub-intent ML model loaded successfully")
        else:
            logger.info("No sub-intent ML model found — using regex fallback")
    return _sub_intent_model


def _get_confidence_model():
    global _confidence_model, _confidence_loaded
    if not _confidence_loaded:
        _confidence_model = load_model("confidence_model")
        _confidence_loaded = True
        if _confidence_model:
            logger.info("Confidence ML model loaded successfully")
        else:
            logger.info("No confidence ML model found — using regex fallback")
    return _confidence_model


def _get_politeness_model():
    global _politeness_model, _politeness_loaded
    if not _politeness_loaded:
        _politeness_model = load_model("politeness_model")
        _politeness_loaded = True
        if _politeness_model:
            logger.info("Politeness ML model loaded successfully")
        else:
            logger.info("No politeness ML model found — using regex fallback")
    return _politeness_model


# ── L8: Sub-Intent Detector ──

SUB_INTENTS = {
    "definition": ["basic_definition", "formal_definition", "intuitive_explanation",
                    "comparison_definition"],
    "how": ["step_by_step", "code_implementation", "tool_usage", "conceptual_steps"],
    "debug": ["error_message", "wrong_output", "environment_issue",
              "syntax_error", "logic_error", "runtime_error", "environment_error"],
    "compare": ["pros_cons", "when_to_use", "performance_comparison"],
    "code": ["syntax_help", "library_usage", "full_example",
             "code_review", "code_optimization"],
    "formula": ["derivation", "intuitive_explanation", "calculation"],
}


def _detect_sub_intent_regex(text_lower: str, intent: str) -> tuple[str, float]:
    """Fallback: regex-based sub-intent detection."""
    subs = SUB_INTENTS.get(intent, [])
    if not subs:
        return "", 0.0

    if intent == "definition":
        if any(w in text_lower for w in ["直覺", "簡單", "白話", "intuition", "simple"]):
            return "intuitive_explanation", 0.7
        elif any(w in text_lower for w in ["正式", "數學", "formal", "rigorous"]):
            return "formal_definition", 0.7
        elif any(w in text_lower for w in ["差別", "不同", "比較", "difference", "compare", "vs"]):
            return "comparison_definition", 0.7
        else:
            return "basic_definition", 0.6
    elif intent == "how":
        if any(w in text_lower for w in ["程式", "code", "python", "import", "implement"]):
            return "code_implementation", 0.7
        elif any(w in text_lower for w in ["步驟", "step", "流程"]):
            return "conceptual_steps", 0.7
        else:
            return "tool_usage", 0.6
    elif intent == "debug":
        if any(w in text_lower for w in ["syntax", "語法", "indentation", "縮排"]):
            return "syntax_error", 0.7
        elif any(w in text_lower for w in ["error", "錯誤", "traceback", "exception"]):
            return "runtime_error", 0.7
        elif any(w in text_lower for w in ["裝", "install", "版本", "version", "環境"]):
            return "environment_error", 0.7
        elif any(w in text_lower for w in ["結果不對", "wrong output", "不下降", "nan"]):
            return "logic_error", 0.7
        else:
            return "runtime_error", 0.5
    elif intent == "compare":
        if any(w in text_lower for w in ["優缺", "pros", "cons", "好壞"]):
            return "pros_cons", 0.7
        elif any(w in text_lower for w in ["何時", "when", "場景", "scenario"]):
            return "when_to_use", 0.7
        else:
            return "performance_comparison", 0.6
    elif intent == "code":
        if any(w in text_lower for w in ["語法", "syntax", "怎麼寫"]):
            return "syntax_help", 0.7
        elif any(w in text_lower for w in ["套件", "library", "模組", "import"]):
            return "library_usage", 0.7
        elif any(w in text_lower for w in ["review", "看看", "問題", "bug"]):
            return "code_review", 0.7
        elif any(w in text_lower for w in ["優化", "快", "效能", "optimize", "fast"]):
            return "code_optimization", 0.7
        else:
            return "full_example", 0.6
    elif intent == "formula":
        if any(w in text_lower for w in ["推導", "derive", "derivation", "展開"]):
            return "derivation", 0.7
        elif any(w in text_lower for w in ["直覺", "白話", "intuition", "meaning", "為什麼"]):
            return "intuitive_explanation", 0.7
        elif any(w in text_lower for w in ["算", "計算", "calculate", "代入"]):
            return "calculation", 0.7
        else:
            return "derivation", 0.5

    return subs[0] if subs else "", 0.5


def sub_intent_detector(ctx: NLPContext) -> NLPContext:
    """L8: Detect sub-intent within the main intent category — ML model with regex fallback."""
    text = ctx.user_message
    text_lower = text.lower()

    # Try ML model first
    model = _get_sub_intent_model()
    if model is not None:
        try:
            # Combine user message with intent for better prediction
            combined = f"{ctx.intent}: {text}"
            prediction = model.predict([combined])[0]
            try:
                decision = model.decision_function([combined])
                import numpy as np
                if decision.ndim > 1:
                    exp_vals = np.exp(decision[0] - np.max(decision[0]))
                    probs = exp_vals / exp_vals.sum()
                    confidence = float(np.max(probs))
                else:
                    confidence = float(min(abs(decision[0]) / 2.0, 1.0))
            except Exception:
                confidence = 0.7

            if confidence > 0.3:
                ctx.sub_intent = prediction
                ctx.sub_intent_confidence = confidence
                return ctx
        except Exception as e:
            logger.warning("Sub-intent ML prediction failed: %s", e)

    # Fallback to regex
    ctx.sub_intent, ctx.sub_intent_confidence = _detect_sub_intent_regex(text_lower, ctx.intent)
    return ctx


# ── L10: Sentiment Scorer ──

def sentiment_scorer(ctx: NLPContext) -> NLPContext:
    """L10: Continuous sentiment score (0=negative, 1=positive) using SnowNLP."""
    SnowNLP = _get_snownlp()
    if SnowNLP is None:
        ctx.sentiment_score = 0.5
        return ctx
    try:
        s = SnowNLP(ctx.user_message)
        ctx.sentiment_score = round(s.sentiments, 3)
    except Exception:
        ctx.sentiment_score = 0.5
    return ctx


# ── L11: Frustration Escalator (enhanced with snownlp sentiment trend) ──

def frustration_escalator(ctx: NLPContext) -> NLPContext:
    """L11: Track frustration level across conversation turns (0-5) with sentiment trend."""
    level = 0

    # Base from emotion
    if ctx.emotion == "frustrated":
        level = 3
    elif ctx.emotion == "confused":
        level = 1

    # Escalate based on sentiment
    if ctx.sentiment_score < 0.1:
        level += 2
    elif ctx.sentiment_score < 0.2:
        level += 1

    # Multi-turn escalation with snownlp sentiment trend
    SnowNLP = _get_snownlp()
    user_msgs = [m.get("content", "") for m in ctx.conversation_history if m.get("role") == "user"]

    if SnowNLP is not None and len(user_msgs) >= 2:
        try:
            sentiments = []
            for msg in user_msgs[-4:]:  # Last 4 messages
                s = SnowNLP(msg)
                sentiments.append(s.sentiments)
            # Check for decreasing sentiment trend
            if len(sentiments) >= 2:
                trend = sentiments[-1] - sentiments[0]
                if trend < -0.3:
                    level += 2  # Strong negative trend
                elif trend < -0.15:
                    level += 1  # Moderate negative trend
        except Exception:
            pass

    # Keyword-based frustration from history (original approach as fallback)
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


# ── L12: Confidence Estimator (ML model with regex fallback) ──

CONFIDENCE_HIGH = [
    r"我覺得", r"我認為", r"應該是", r"我確定", r"我知道",
    r"i think", r"i believe", r"i'm sure", r"definitely",
]
CONFIDENCE_LOW = [
    r"不確定", r"可能", r"也許", r"不知道", r"大概",
    r"not sure", r"maybe", r"perhaps", r"i guess",
]

_CONFIDENCE_SCORE_MAP = {"high": 0.85, "medium": 0.5, "low": 0.15}


def confidence_estimator(ctx: NLPContext) -> NLPContext:
    """L12: Estimate student confidence level (0-1) — ML model with regex fallback."""
    text = ctx.user_message
    text_lower = text.lower()

    # Try ML model first
    model = _get_confidence_model()
    if model is not None:
        try:
            prediction = model.predict([text])[0]
            try:
                proba = model.predict_proba([text])[0]
                ml_confidence = float(max(proba))
            except Exception:
                ml_confidence = 0.7

            if ml_confidence > 0.4:
                base_score = _CONFIDENCE_SCORE_MAP.get(prediction, 0.5)
                # Blend with sentiment
                sentiment_component = ctx.sentiment_score * 0.2
                ctx.confidence_level = min(base_score * 0.8 + sentiment_component, 1.0)
                return ctx
        except Exception as e:
            logger.warning("Confidence ML prediction failed: %s", e)

    # Fallback to regex
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


# ── L14: Politeness Detector (ML model with regex+snownlp fallback) ──

POLITE_SIGNALS = [
    r"請", r"麻煩", r"謝謝", r"感謝", r"不好意思", r"打擾",
    r"please", r"thank", r"sorry", r"appreciate", r"kindly",
]

_POLITENESS_SCORE_MAP = {"polite": 0.85, "neutral": 0.5, "direct": 0.2}


def politeness_detector(ctx: NLPContext) -> NLPContext:
    """L14: Detect politeness level (0-1) — ML model with regex+snownlp fallback."""
    text = ctx.user_message
    text_lower = text.lower()

    # Try ML model first
    model = _get_politeness_model()
    if model is not None:
        try:
            prediction = model.predict([text])[0]
            try:
                proba = model.predict_proba([text])[0]
                ml_confidence = float(max(proba))
            except Exception:
                ml_confidence = 0.7

            if ml_confidence > 0.4:
                base_score = _POLITENESS_SCORE_MAP.get(prediction, 0.5)
                # Blend with snownlp sentiment
                sentiment_component = ctx.sentiment_score * 0.2
                ctx.politeness_score = min(base_score * 0.8 + sentiment_component, 1.0)
                return ctx
        except Exception as e:
            logger.warning("Politeness ML prediction failed: %s", e)

    # Fallback to regex + snownlp
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

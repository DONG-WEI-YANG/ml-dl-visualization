"""Layer 2: Emotion Detection — ML classifier with regex fallback."""

import re
import logging
from .pipeline import NLPContext
from .trainer import load_model

logger = logging.getLogger(__name__)

# ── Cached model ──
_emotion_model = None
_model_loaded = False


def _get_model():
    """Lazy-load the trained emotion model."""
    global _emotion_model, _model_loaded
    if not _model_loaded:
        _emotion_model = load_model("emotion_model")
        _model_loaded = True
        if _emotion_model:
            logger.info("Emotion ML model loaded successfully")
        else:
            logger.info("No emotion ML model found — using regex fallback")
    return _emotion_model


def reload_model():
    """Force reload after retraining."""
    global _model_loaded
    _model_loaded = False


# ── Regex fallback patterns ──

EMOTION_PATTERNS = {
    "frustrated": [
        r"不懂", r"看不懂", r"搞不懂", r"完全不", r"到底", r"一直",
        r"做不出", r"跑不出", r"試了很多", r"試了好久", r"放棄",
        r"很煩", r"好難", r"太難", r"超難", r"崩潰", r"頭痛",
        r"don't understand", r"stuck", r"frustrated", r"give up",
        r"impossible", r"too hard", r"so confused",
    ],
    "confused": [
        r"不確定", r"搞混", r"混淆", r"分不清", r"不太懂", r"有點", r"不太理解",
        r"疑惑", r"困惑", r"模糊", r"不清楚", r"哪裡錯",
        r"confused", r"not sure", r"unclear", r"lost", r"which one",
    ],
    "curious": [
        r"好奇", r"有趣", r"想了解", r"想知道", r"想問", r"請問",
        r"為什麼會", r"怎麼可能", r"居然", r"原來",
        r"curious", r"interesting", r"wonder", r"fascinating",
    ],
    "confident": [
        r"我覺得", r"我認為", r"應該是", r"我理解", r"我知道",
        r"確認一下", r"對不對", r"是否正確", r"驗證",
        r"i think", r"i believe", r"verify", r"confirm", r"correct\?",
    ],
}

URGENCY_PATTERNS = {
    "high": [
        r"急", r"趕", r"明天", r"今天", r"馬上", r"快", r"來不及",
        r"deadline", r"due", r"urgent", r"asap", r"tonight", r"tomorrow",
    ],
    "low": [
        r"順便", r"閒聊", r"好奇而已", r"隨便問", r"有空",
        r"just wondering", r"no rush", r"by the way",
    ],
}


def _detect_emotion_regex(text_lower: str) -> str:
    """Fallback: regex-based emotion detection."""
    emotion_scores: dict[str, int] = {}
    for emotion, patterns in EMOTION_PATTERNS.items():
        count = sum(1 for p in patterns if re.search(p, text_lower))
        if count > 0:
            emotion_scores[emotion] = count

    if emotion_scores:
        return max(emotion_scores, key=emotion_scores.get)
    return "neutral"


def _detect_emotion_ml(text: str) -> tuple[str, float]:
    """ML-based emotion detection using trained TF-IDF + LogisticRegression."""
    model = _get_model()
    if model is None:
        return None, 0.0

    prediction = model.predict([text])[0]

    # LogisticRegression supports predict_proba
    try:
        proba = model.predict_proba([text])[0]
        confidence = float(max(proba))
    except Exception:
        confidence = 0.7

    return prediction, confidence


def detect_emotion(ctx: NLPContext) -> NLPContext:
    """Detect student emotional state — ML model with regex fallback."""
    text = ctx.user_message
    text_lower = text.lower()

    # Try ML model first
    ml_emotion, ml_confidence = _detect_emotion_ml(text)

    if ml_emotion is not None and ml_confidence > 0.35:
        ctx.emotion = ml_emotion
    else:
        # Fallback to regex
        ctx.emotion = _detect_emotion_regex(text_lower)

    # Multi-turn frustration escalation
    if ctx.turn_count > 3 and ctx.emotion in ("confused", "neutral"):
        ctx.emotion = "confused"

    # Detect urgency (always regex — simple pattern matching is sufficient)
    for level, patterns in URGENCY_PATTERNS.items():
        if any(re.search(p, text_lower) for p in patterns):
            ctx.urgency = level
            break

    return ctx

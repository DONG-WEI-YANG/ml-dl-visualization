"""Layer 1: Intent Detection — ML classifier with regex fallback."""

import re
import logging
from .pipeline import NLPContext
from .trainer import load_model

logger = logging.getLogger(__name__)

# ── Cached model ──
_intent_model = None
_model_loaded = False


def _get_model():
    """Lazy-load the trained intent model."""
    global _intent_model, _model_loaded
    if not _model_loaded:
        _intent_model = load_model("intent_model")
        _model_loaded = True
        if _intent_model:
            logger.info("Intent ML model loaded successfully")
        else:
            logger.info("No intent ML model found — using regex fallback")
    return _intent_model


def reload_model():
    """Force reload after retraining."""
    global _model_loaded
    _model_loaded = False


# ── Regex fallback patterns ──

INTENT_PATTERNS = {
    "definition": [
        (r"什麼是", 1.0), (r"何謂", 1.0), (r"定義", 1.0), (r"是什麼", 0.9),
        (r"意思", 0.8), (r"含義", 0.9), (r"概念", 0.7),
        (r"what is", 1.0), (r"what are", 1.0), (r"define", 1.0), (r"meaning of", 0.9),
    ],
    "how": [
        (r"怎麼做", 1.0), (r"如何實作", 1.0), (r"怎樣寫", 0.9), (r"步驟", 0.9),
        (r"方法", 0.7), (r"流程", 0.8), (r"實現", 0.8), (r"實作", 0.9),
        (r"how to", 1.0), (r"how do", 0.9), (r"how can", 0.9), (r"implement", 0.9),
    ],
    "why": [
        (r"為什麼", 1.0), (r"為何", 1.0), (r"原因", 0.9), (r"理由", 0.9),
        (r"目的", 0.8), (r"動機", 0.8),
        (r"why", 1.0), (r"reason", 0.9), (r"purpose", 0.8),
    ],
    "compare": [
        (r"差別", 1.0), (r"差異", 1.0), (r"比較", 1.0), (r"不同", 0.8),
        (r"區別", 1.0), (r"優劣", 0.9), (r"優缺點", 0.9), (r"選擇哪個", 0.9),
        (r"vs", 1.0), (r"versus", 1.0), (r"difference", 1.0), (r"compare", 1.0),
    ],
    "example": [
        (r"舉例", 1.0), (r"例子", 1.0), (r"範例", 1.0), (r"示範", 0.9),
        (r"實例", 0.9), (r"案例", 0.8),
        (r"example", 1.0), (r"show me", 0.9), (r"demonstrate", 0.9),
    ],
    "debug": [
        (r"錯誤", 0.9), (r"報錯", 1.0), (r"失敗", 0.8), (r"異常", 0.9),
        (r"error", 1.0), (r"bug", 1.0), (r"fail", 0.9), (r"crash", 1.0),
        (r"traceback", 1.0), (r"not working", 0.9), (r"exception", 1.0),
    ],
    "formula": [
        (r"公式", 1.0), (r"數學", 0.8), (r"推導", 1.0), (r"計算", 0.8),
        (r"方程", 1.0), (r"證明", 0.9),
        (r"formula", 1.0), (r"equation", 1.0), (r"derive", 1.0), (r"proof", 0.9),
    ],
    "code": [
        (r"程式碼", 1.0), (r"語法", 0.9), (r"函數", 0.8), (r"函式", 0.8),
        (r"import", 0.9), (r"呼叫", 0.7),
        (r"code", 1.0), (r"syntax", 0.9), (r"library", 0.8), (r"package", 0.7),
    ],
    "parameter": [
        (r"參數", 1.0), (r"超參數", 1.0), (r"調參", 1.0), (r"怎麼調", 0.9),
        (r"怎麼設", 0.9), (r"預設值", 0.8),
        (r"parameter", 1.0), (r"hyperparameter", 1.0), (r"tuning", 0.9),
    ],
    "performance": [
        (r"準確率", 1.0), (r"效能", 0.8), (r"評估", 0.8), (r"指標", 0.9),
        (r"表現", 0.7), (r"改善", 0.8), (r"提升", 0.7),
        (r"accuracy", 1.0), (r"metric", 0.9), (r"f1", 1.0), (r"auc", 1.0),
    ],
    "data": [
        (r"資料集", 1.0), (r"前處理", 1.0), (r"清理", 0.9), (r"缺失值", 1.0),
        (r"特徵工程", 1.0),
        (r"dataset", 1.0), (r"preprocess", 1.0), (r"feature engineer", 1.0),
        (r"normalize", 0.9), (r"missing", 0.9),
    ],
    "visualization": [
        (r"視覺化", 1.0), (r"圖表", 0.9), (r"畫圖", 1.0), (r"繪製", 0.9),
        (r"怎麼畫", 1.0), (r"畫.*曲線", 1.0), (r"畫.*圖", 1.0),
        (r"曲線圖", 1.0), (r"散佈圖", 1.0), (r"熱力圖", 1.0),
        (r"plot", 1.0), (r"visualize", 1.0), (r"matplotlib", 1.0),
    ],
    "intuition": [
        (r"直覺", 1.0), (r"原理", 0.9), (r"背後", 0.8), (r"本質", 0.9),
        (r"核心", 0.7), (r"思路", 0.8),
        (r"intuition", 1.0), (r"insight", 0.9), (r"idea behind", 1.0),
    ],
    "application": [
        (r"應用", 1.0), (r"實際", 0.8), (r"場景", 0.9), (r"用途", 0.9),
        (r"產業", 0.9), (r"業界", 0.9),
        (r"application", 1.0), (r"real.world", 1.0), (r"use case", 1.0),
    ],
    "prerequisite": [
        (r"先備", 1.0), (r"需要先", 0.9), (r"基礎", 0.8), (r"先學", 1.0),
        (r"prerequisite", 1.0), (r"background", 0.8), (r"foundation", 0.8),
    ],
    "summary": [
        (r"總結", 1.0), (r"重點", 1.0), (r"摘要", 1.0), (r"複習", 1.0),
        (r"回顧", 0.9), (r"歸納", 0.9),
        (r"summary", 1.0), (r"review", 0.9), (r"recap", 1.0), (r"key point", 1.0),
    ],
    "troubleshoot": [
        (r"跑不動", 1.0), (r"裝不了", 1.0), (r"安裝", 0.9), (r"環境", 0.8),
        (r"版本", 0.8), (r"相容", 0.9), (r"衝突", 0.9),
        (r"install", 1.0), (r"setup", 0.9), (r"version", 0.8), (r"dependency", 0.9),
    ],
    "deeper": [
        (r"更深入", 1.0), (r"進階", 1.0), (r"延伸", 0.9), (r"詳細", 0.8),
        (r"補充", 0.8), (r"細節", 0.7),
        (r"deeper", 1.0), (r"advanced", 0.9), (r"elaborate", 0.9), (r"further", 0.8),
    ],
}

PRIORITY_BOOST = {
    "intuition": 0.3, "visualization": 0.3, "troubleshoot": 0.3,
    "formula": 0.2, "code": 0.2, "parameter": 0.2, "deeper": 0.2,
}


def _detect_intent_regex(text_lower: str) -> tuple[str, float]:
    """Fallback: regex-based intent detection."""
    scores: dict[str, float] = {}
    for intent, patterns in INTENT_PATTERNS.items():
        total = 0.0
        for pat, weight in patterns:
            if re.search(pat, text_lower):
                total += weight
        if total > 0:
            scores[intent] = total + PRIORITY_BOOST.get(intent, 0)

    if scores:
        best = max(scores, key=scores.get)
        max_possible = max(sum(w for _, w in INTENT_PATTERNS[best]), 1)
        confidence = min(scores[best] / max_possible, 1.0)
        return best, confidence
    return "general", 0.0


def _detect_intent_ml(text: str) -> tuple[str, float]:
    """ML-based intent detection using trained TF-IDF + LinearSVC."""
    model = _get_model()
    if model is None:
        return None, 0.0

    prediction = model.predict([text])[0]

    # Get confidence from decision function (LinearSVC)
    try:
        decision = model.decision_function([text])
        # For multi-class, decision_function returns shape (1, n_classes)
        import numpy as np
        if decision.ndim > 1:
            # Softmax-like normalization of decision values
            exp_vals = np.exp(decision[0] - np.max(decision[0]))
            probs = exp_vals / exp_vals.sum()
            confidence = float(np.max(probs))
        else:
            confidence = float(min(abs(decision[0]) / 2.0, 1.0))
    except Exception:
        confidence = 0.7  # Default confidence for ML prediction

    return prediction, confidence


def detect_intent(ctx: NLPContext) -> NLPContext:
    """Classify question intent — ML model with regex fallback."""
    text = ctx.user_message
    text_lower = text.lower()

    # Try ML model first
    ml_intent, ml_confidence = _detect_intent_ml(text)

    if ml_intent is not None and ml_confidence > 0.3:
        ctx.intent = ml_intent
        ctx.intent_confidence = ml_confidence
    else:
        # Fallback to regex
        ctx.intent, ctx.intent_confidence = _detect_intent_regex(text_lower)

    return ctx

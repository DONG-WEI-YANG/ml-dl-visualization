"""Group C: Student Level Assessment Layers (L15-20).

L15 DifficultyAssessor — textstat + jieba (enhanced from existing)
L16 VocabularyLevelScorer — jieba + domain term bank
L17 TechnicalFluencyScorer — POS analysis
L18 LearningStyleDetector — pattern-based
L19 MisconceptionDetector — rapidfuzz + misconception bank
L20 KnowledgeGapDetector — concept coverage analysis
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_textstat = None
_rapidfuzz = None


def _get_textstat():
    global _textstat
    if _textstat is None:
        import textstat
        _textstat = textstat
    return _textstat


def _get_rapidfuzz():
    global _rapidfuzz
    if _rapidfuzz is None:
        from rapidfuzz import fuzz
        _rapidfuzz = fuzz
    return _rapidfuzz


# ── Domain term bank ──

BEGINNER_TERMS = {"變數", "迴圈", "函數", "列表", "字典", "variable", "loop", "function", "list", "print"}

INTERMEDIATE_TERMS = {
    "過擬合", "欠擬合", "交叉驗證", "損失函數", "梯度下降", "學習率",
    "決策邊界", "特徵工程", "標準化", "SVM", "隨機森林",
    "overfitting", "underfitting", "cross-validation", "loss function",
    "gradient descent", "learning rate", "decision boundary", "random forest",
}

ADVANCED_TERMS = {
    "反向傳播", "注意力機制", "Transformer", "遷移學習", "批次正規化",
    "SHAP", "Grad-CAM", "嵌入空間", "自注意力", "位置編碼",
    "backpropagation", "attention mechanism", "transfer learning",
    "batch normalization", "positional encoding", "ablation", "SOTA",
}

# ── Common misconceptions in ML/DL ──

MISCONCEPTIONS = {
    "accuracy 越高越好": "在類別不平衡時，accuracy 可能誤導。應搭配 F1、AUC 等指標。",
    "更多特徵一定更好": "過多特徵可能導致維度災難和過擬合，需做特徵選擇。",
    "深度學習一定比傳統ML好": "資料少或結構化資料上，傳統 ML（如 XGBoost）常常更好。",
    "訓練損失越低模型越好": "訓練損失過低可能是過擬合，需看驗證損失。",
    "學習率越小越好": "學習率太小會收斂極慢或卡在局部最小值。",
    "batch size 越大越好": "過大的 batch size 可能導致泛化能力下降。",
    "dropout 永遠有幫助": "在資料量充足且模型不過擬合時，dropout 可能降低效能。",
    "CNN 只能用在影像": "CNN 也可用於文本分類、時序資料等一維序列。",
    "RNN 已經被淘汰": "RNN/LSTM 在某些小資料序列任務上仍有優勢。",
    "正則化就是 L2": "正則化包含 L1、L2、Dropout、Early Stopping、資料增強等多種技術。",
}

# ── Week-concept mapping for knowledge gap detection ──

WEEK_CONCEPTS = {
    1: ["Python", "NumPy", "Pandas", "Jupyter"],
    2: ["EDA", "Matplotlib", "Seaborn", "Plotly", "散佈圖", "直方圖"],
    3: ["監督式學習", "訓練集", "測試集", "交叉驗證", "過擬合", "欠擬合"],
    4: ["線性回歸", "損失函數", "MSE", "梯度下降", "學習率"],
    5: ["邏輯迴歸", "Sigmoid", "決策邊界", "ROC", "AUC", "F1"],
    6: ["SVM", "核方法", "RBF", "間隔最大化", "支撐向量"],
    7: ["決策樹", "隨機森林", "GBDT", "Bagging", "Boosting"],
    8: ["特徵重要度", "SHAP", "排列重要度", "可解釋性"],
    9: ["特徵工程", "Pipeline", "One-Hot", "StandardScaler"],
    10: ["超參數", "GridSearch", "RandomSearch", "學習曲線"],
    11: ["神經網路", "激活函數", "ReLU", "Dropout", "BatchNorm"],
    12: ["CNN", "卷積", "池化", "特徵圖", "Grad-CAM"],
    13: ["RNN", "LSTM", "GRU", "Transformer", "注意力機制"],
    14: ["學習率排程", "早停", "資料增強", "訓練曲線"],
    15: ["混淆矩陣", "公平性", "偏誤", "穩健性"],
    16: ["MLOps", "MLflow", "模型版本", "資料漂移"],
    17: ["LLM", "嵌入", "RAG", "提示工程"],
    18: ["專題報告", "可重現性", "倫理"],
}


# ── L15: Difficulty Assessor (enhanced) ──

def difficulty_assessor(ctx: NLPContext) -> NLPContext:
    """L15: Assess student difficulty level from vocabulary and text complexity."""
    text_lower = ctx.user_message.lower()

    # Count domain terms by level
    beginner_count = sum(1 for t in BEGINNER_TERMS if t.lower() in text_lower)
    intermediate_count = sum(1 for t in INTERMEDIATE_TERMS if t.lower() in text_lower)
    advanced_count = sum(1 for t in ADVANCED_TERMS if t.lower() in text_lower)

    ctx.uses_technical_terms = (intermediate_count + advanced_count) > 0

    # Determine level
    if advanced_count >= 2 or (advanced_count >= 1 and intermediate_count >= 2):
        ctx.student_level = "advanced"
    elif intermediate_count >= 2 or (intermediate_count >= 1 and beginner_count >= 1):
        ctx.student_level = "intermediate"
    else:
        ctx.student_level = "beginner"

    # Question complexity
    msg_len = len(ctx.user_message)
    q_marks = ctx.user_message.count("\uff1f") + ctx.user_message.count("?")

    if q_marks > 1 or ctx.has_code or msg_len > 200:
        ctx.question_complexity = "complex"
    elif msg_len > 80 or (intermediate_count + advanced_count) >= 2:
        ctx.question_complexity = "moderate"
    else:
        ctx.question_complexity = "simple"

    return ctx


# ── L16: Vocabulary Level Scorer ──

def vocabulary_level_scorer(ctx: NLPContext) -> NLPContext:
    """L16: Score vocabulary richness based on domain term usage."""
    text_lower = ctx.user_message.lower()
    all_terms = BEGINNER_TERMS | INTERMEDIATE_TERMS | ADVANCED_TERMS
    used = [t for t in all_terms if t.lower() in text_lower]

    if not used:
        ctx.vocabulary_score = 0.0
        return ctx

    # Weight: advanced=3, intermediate=2, beginner=1
    score = 0
    for t in used:
        if t in ADVANCED_TERMS or t.lower() in {x.lower() for x in ADVANCED_TERMS}:
            score += 3
        elif t in INTERMEDIATE_TERMS or t.lower() in {x.lower() for x in INTERMEDIATE_TERMS}:
            score += 2
        else:
            score += 1

    # Normalize to 0-1 (max realistic score ~15)
    ctx.vocabulary_score = min(score / 15.0, 1.0)
    return ctx


# ── L17: Technical Fluency Scorer ──

def technical_fluency_scorer(ctx: NLPContext) -> NLPContext:
    """L17: Score technical expression fluency using POS tags."""
    if not ctx.pos_tags:
        ctx.technical_fluency = 0.0
        return ctx

    # Technical POS: nouns (n, eng), verbs related to tech
    tech_pos = {"eng", "n", "nz", "nr", "vn"}
    tech_count = sum(1 for _, flag in ctx.pos_tags if flag in tech_pos)
    total = len(ctx.pos_tags)

    if total == 0:
        ctx.technical_fluency = 0.0
    else:
        ctx.technical_fluency = min(tech_count / total, 1.0)

    return ctx


# ── L18: Learning Style Detector ──

VISUAL_SIGNALS = ["圖", "畫", "視覺", "看", "顯示", "plot", "chart", "graph", "visual", "show"]
TEXTUAL_SIGNALS = ["解釋", "說明", "描述", "文字", "explain", "describe", "text", "read"]
PRACTICAL_SIGNALS = ["實作", "練習", "跑", "程式", "code", "implement", "try", "run", "practice"]


def learning_style_detector(ctx: NLPContext) -> NLPContext:
    """L18: Detect preferred learning style (visual/textual/practical/balanced)."""
    text_lower = ctx.user_message.lower()

    v = sum(1 for s in VISUAL_SIGNALS if s in text_lower)
    t = sum(1 for s in TEXTUAL_SIGNALS if s in text_lower)
    p = sum(1 for s in PRACTICAL_SIGNALS if s in text_lower)

    total = v + t + p
    if total == 0:
        ctx.learning_style = "balanced"
    elif v > t and v > p:
        ctx.learning_style = "visual"
    elif t > v and t > p:
        ctx.learning_style = "textual"
    elif p > v and p > t:
        ctx.learning_style = "practical"
    else:
        ctx.learning_style = "balanced"

    return ctx


# ── L19: Misconception Detector ──

def misconception_detector(ctx: NLPContext) -> NLPContext:
    """L19: Detect common ML/DL misconceptions using fuzzy matching."""
    fuzz = _get_rapidfuzz()
    text = ctx.user_message
    ctx.misconceptions = []

    for trigger, correction in MISCONCEPTIONS.items():
        score = fuzz.partial_ratio(trigger, text)
        if score > 75:
            ctx.misconceptions.append(f"\u26a0\ufe0f \u5e38\u898b\u8ff7\u601d\uff1a\u300c{trigger}\u300d\u2014 {correction}")

    return ctx


# ── L20: Knowledge Gap Detector ──

def knowledge_gap_detector(ctx: NLPContext) -> NLPContext:
    """L20: Detect knowledge gaps by comparing question with prerequisite concepts."""
    current_week = ctx.week
    text_lower = ctx.user_message.lower()

    # Check if student is asking about concepts from earlier weeks
    gaps = []
    for w in range(1, current_week):
        concepts = WEEK_CONCEPTS.get(w, [])
        for concept in concepts:
            if concept.lower() in text_lower:
                # If asking basic questions about prerequisite concepts, might be a gap
                if ctx.intent in ("definition", "how", "prerequisite") and ctx.student_level == "beginner":
                    gaps.append(f"\u7b2c{w}\u9031\uff1a{concept}")

    ctx.knowledge_gaps = gaps[:3]  # Top 3 gaps

    # Track known concepts (mentioned with confidence)
    current_concepts = WEEK_CONCEPTS.get(current_week, [])
    known = [c for c in current_concepts if c.lower() in text_lower and ctx.confidence_level > 0.6]
    ctx.known_concepts = known

    # Unknown = current week concepts NOT mentioned
    ctx.unknown_concepts = [c for c in current_concepts if c not in known]

    return ctx


# ── Public aliases (used by __init__.py FULL_PIPELINE) ──

score_vocabulary = vocabulary_level_scorer
score_technical_fluency = technical_fluency_scorer
detect_learning_style = learning_style_detector
detect_misconceptions = misconception_detector
detect_knowledge_gaps = knowledge_gap_detector

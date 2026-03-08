"""Group C: Student Level Assessment Layers (L15-20).

L15 DifficultyAssessor — textstat + jieba (enhanced from existing)
L16 VocabularyLevelScorer — jieba TF-IDF + wordfreq rarity
L17 TechnicalFluencyScorer — jieba.posseg POS distribution
L18 LearningStyleDetector — ML model + pattern fallback
L19 MisconceptionDetector — rapidfuzz + misconception bank
L20 KnowledgeGapDetector — jieba keyword extraction + concept-prerequisite mapping
"""

import re
import logging
from .pipeline import NLPContext
from .trainer import load_model

logger = logging.getLogger(__name__)

_textstat = None
_rapidfuzz = None

# ── Cached ML model ──
_learning_style_model = None
_learning_style_loaded = False


def _get_textstat():
    global _textstat
    if _textstat is None:
        try:
            import textstat
            _textstat = textstat
        except ImportError:
            logger.warning("textstat not available")
            _textstat = False
    return _textstat if _textstat is not False else None


def _get_rapidfuzz():
    global _rapidfuzz
    if _rapidfuzz is None:
        try:
            from rapidfuzz import fuzz
            _rapidfuzz = fuzz
        except ImportError:
            logger.warning("rapidfuzz not available")
            _rapidfuzz = False
    return _rapidfuzz if _rapidfuzz is not False else None


def _get_learning_style_model():
    global _learning_style_model, _learning_style_loaded
    if not _learning_style_loaded:
        _learning_style_model = load_model("learning_style_model")
        _learning_style_loaded = True
        if _learning_style_model:
            logger.info("Learning style ML model loaded successfully")
        else:
            logger.info("No learning style ML model found — using regex fallback")
    return _learning_style_model


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

# ── Concept prerequisite mapping for knowledge gap detection ──

CONCEPT_PREREQUISITES = {
    "梯度下降": ["微積分", "偏微分", "損失函數"],
    "反向傳播": ["梯度下降", "鏈式法則", "神經網路"],
    "CNN": ["神經網路", "卷積", "激活函數"],
    "RNN": ["神經網路", "序列資料", "激活函數"],
    "LSTM": ["RNN", "梯度消失"],
    "Transformer": ["注意力機制", "嵌入", "位置編碼"],
    "注意力機制": ["矩陣乘法", "softmax", "嵌入"],
    "SVM": ["決策邊界", "超平面", "核方法"],
    "隨機森林": ["決策樹", "Bagging", "集成學習"],
    "GBDT": ["決策樹", "Boosting", "梯度下降"],
    "交叉驗證": ["訓練集", "測試集", "過擬合"],
    "正則化": ["過擬合", "損失函數", "權重"],
    "特徵工程": ["標準化", "One-Hot", "特徵選擇"],
    "遷移學習": ["CNN", "預訓練模型", "微調"],
    "SHAP": ["特徵重要度", "決策樹", "可解釋性"],
    "gradient descent": ["calculus", "partial derivative", "loss function"],
    "backpropagation": ["gradient descent", "chain rule", "neural network"],
    "overfitting": ["training set", "test set", "bias-variance"],
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


# ── L16: Vocabulary Level Scorer (jieba TF-IDF + wordfreq) ──

def vocabulary_level_scorer(ctx: NLPContext) -> NLPContext:
    """L16: Score vocabulary richness using jieba TF-IDF + wordfreq rarity."""
    text = ctx.user_message

    # Try jieba TF-IDF keyword extraction + wordfreq rarity scoring
    try:
        import jieba.analyse
        from wordfreq import word_frequency

        # Extract keywords with TF-IDF weights
        tfidf_kw = jieba.analyse.extract_tags(text, topK=15, withWeight=True)

        if not tfidf_kw:
            ctx.vocabulary_score = 0.0
            return ctx

        # Compute rarity-weighted score
        rarity_scores = []
        for kw, weight in tfidf_kw:
            # Get word frequency (lower = rarer = more advanced)
            freq_zh = word_frequency(kw, 'zh')
            freq_en = word_frequency(kw, 'en')
            freq = max(freq_zh, freq_en)

            # Convert frequency to rarity score (0-1)
            if freq == 0:
                rarity = 1.0  # Unknown word = very rare
            elif freq < 1e-6:
                rarity = 0.9
            elif freq < 1e-5:
                rarity = 0.7
            elif freq < 1e-4:
                rarity = 0.5
            elif freq < 1e-3:
                rarity = 0.3
            else:
                rarity = 0.1  # Common word

            rarity_scores.append(rarity * weight)

        # Average rarity weighted by TF-IDF importance
        total_weight = sum(w for _, w in tfidf_kw)
        if total_weight > 0:
            vocab_score = sum(rarity_scores) / total_weight
        else:
            vocab_score = 0.0

        # Also factor in domain term usage
        text_lower = text.lower()
        all_terms = BEGINNER_TERMS | INTERMEDIATE_TERMS | ADVANCED_TERMS
        used_terms = [t for t in all_terms if t.lower() in text_lower]
        domain_bonus = 0
        for t in used_terms:
            if t in ADVANCED_TERMS or t.lower() in {x.lower() for x in ADVANCED_TERMS}:
                domain_bonus += 0.06
            elif t in INTERMEDIATE_TERMS or t.lower() in {x.lower() for x in INTERMEDIATE_TERMS}:
                domain_bonus += 0.04
            else:
                domain_bonus += 0.02

        ctx.vocabulary_score = min(vocab_score + domain_bonus, 1.0)
        return ctx

    except ImportError:
        logger.warning("jieba or wordfreq not available — using basic vocabulary scoring")

    # Fallback: basic domain term counting
    text_lower = text.lower()
    all_terms = BEGINNER_TERMS | INTERMEDIATE_TERMS | ADVANCED_TERMS
    used = [t for t in all_terms if t.lower() in text_lower]

    if not used:
        ctx.vocabulary_score = 0.0
        return ctx

    score = 0
    for t in used:
        if t in ADVANCED_TERMS or t.lower() in {x.lower() for x in ADVANCED_TERMS}:
            score += 3
        elif t in INTERMEDIATE_TERMS or t.lower() in {x.lower() for x in INTERMEDIATE_TERMS}:
            score += 2
        else:
            score += 1

    ctx.vocabulary_score = min(score / 15.0, 1.0)
    return ctx


# ── L17: Technical Fluency Scorer (jieba.posseg POS distribution) ──

def technical_fluency_scorer(ctx: NLPContext) -> NLPContext:
    """L17: Score technical expression fluency using jieba POS distribution."""
    text = ctx.user_message

    # Try jieba.posseg for detailed POS analysis
    try:
        import jieba.posseg as pseg

        pairs = list(pseg.lcut(text))
        if not pairs:
            ctx.technical_fluency = 0.0
            return ctx

        total = len(pairs)
        # Technical POS: nouns (n, nr, ns, nt, nz), English terms (eng)
        tech_tags = {"n", "nr", "ns", "nt", "nz", "eng"}
        tech_count = sum(1 for word, flag in pairs if flag in tech_tags)

        # Also count English-looking tokens (code, API names)
        eng_count = sum(1 for word, _ in pairs if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', word) and len(word) > 1)
        tech_count += eng_count

        if total == 0:
            ctx.technical_fluency = 0.0
        else:
            ctx.technical_fluency = min(tech_count / total, 1.0)

        return ctx

    except ImportError:
        logger.warning("jieba.posseg not available — using basic POS-based scoring")

    # Fallback: use pre-computed POS tags
    if not ctx.pos_tags:
        ctx.technical_fluency = 0.0
        return ctx

    tech_pos = {"eng", "n", "nz", "nr", "vn"}
    tech_count = sum(1 for _, flag in ctx.pos_tags if flag in tech_pos)
    total = len(ctx.pos_tags)

    if total == 0:
        ctx.technical_fluency = 0.0
    else:
        ctx.technical_fluency = min(tech_count / total, 1.0)

    return ctx


# ── L18: Learning Style Detector (ML model with regex fallback) ──

VISUAL_SIGNALS = ["圖", "畫", "視覺", "看", "顯示", "plot", "chart", "graph", "visual", "show"]
TEXTUAL_SIGNALS = ["解釋", "說明", "描述", "文字", "explain", "describe", "text", "read"]
PRACTICAL_SIGNALS = ["實作", "練習", "跑", "程式", "code", "implement", "try", "run", "practice"]


def learning_style_detector(ctx: NLPContext) -> NLPContext:
    """L18: Detect preferred learning style — ML model with regex fallback."""
    text = ctx.user_message
    text_lower = text.lower()

    # Try ML model first
    model = _get_learning_style_model()
    if model is not None:
        try:
            prediction = model.predict([text])[0]
            try:
                decision = model.decision_function([text])
                import numpy as np
                if decision.ndim > 1:
                    exp_vals = np.exp(decision[0] - np.max(decision[0]))
                    probs = exp_vals / exp_vals.sum()
                    confidence = float(np.max(probs))
                else:
                    confidence = float(min(abs(decision[0]) / 2.0, 1.0))
            except Exception:
                confidence = 0.7

            if confidence > 0.35:
                ctx.learning_style = prediction
                return ctx
        except Exception as e:
            logger.warning("Learning style ML prediction failed: %s", e)

    # Fallback to regex
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
    if fuzz is None:
        ctx.misconceptions = []
        return ctx

    text = ctx.user_message
    ctx.misconceptions = []

    for trigger, correction in MISCONCEPTIONS.items():
        score = fuzz.partial_ratio(trigger, text)
        if score > 75:
            ctx.misconceptions.append(f"\u26a0\ufe0f \u5e38\u898b\u8ff7\u601d\uff1a\u300c{trigger}\u300d\u2014 {correction}")

    return ctx


# ── L20: Knowledge Gap Detector (jieba keyword extraction + concept-prerequisite mapping) ──

def knowledge_gap_detector(ctx: NLPContext) -> NLPContext:
    """L20: Detect knowledge gaps using jieba keyword extraction + prerequisite mapping."""
    current_week = ctx.week
    text = ctx.user_message
    text_lower = text.lower()

    gaps = []

    # Use jieba keyword extraction to identify what the student is asking about
    try:
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(text, topK=10)
    except ImportError:
        keywords = []

    # Check prerequisite concepts for detected domain concepts
    for concept in ctx.domain_concepts:
        clean = concept.split("(")[0].strip()
        prereqs = CONCEPT_PREREQUISITES.get(clean, [])
        for prereq in prereqs:
            # If the student is asking basic questions about a concept with prerequisites
            if ctx.intent in ("definition", "how", "prerequisite") and ctx.student_level == "beginner":
                # Check if prereq is mentioned (student might be lacking it)
                if prereq.lower() not in text_lower:
                    gaps.append(f"先備概念：{prereq}（需要理解{clean}）")

    # Also check jieba-extracted keywords against earlier weeks
    for kw in keywords:
        for w in range(1, current_week):
            concepts = WEEK_CONCEPTS.get(w, [])
            for concept in concepts:
                if concept.lower() in kw.lower() or kw.lower() in concept.lower():
                    if ctx.intent in ("definition", "how", "prerequisite") and ctx.student_level == "beginner":
                        gap_str = f"第{w}週：{concept}"
                        if gap_str not in gaps:
                            gaps.append(gap_str)

    # Legacy: Check if student is asking about concepts from earlier weeks
    for w in range(1, current_week):
        concepts = WEEK_CONCEPTS.get(w, [])
        for concept in concepts:
            if concept.lower() in text_lower:
                if ctx.intent in ("definition", "how", "prerequisite") and ctx.student_level == "beginner":
                    gap_str = f"第{w}週：{concept}"
                    if gap_str not in gaps:
                        gaps.append(gap_str)

    ctx.knowledge_gaps = gaps[:5]  # Top 5 gaps

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

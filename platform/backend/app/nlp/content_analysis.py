"""Group D: Content Analysis Layers (L21-27).

L21 KeywordExtractor — jieba.analyse TF-IDF + TextRank
L22 DomainConceptMatcher — rapidfuzz + concept map
L23 NamedEntityRecognizer — jieba + custom dict
L24 CodeBlockDetector — regex + heuristics
L25 MathExpressionDetector — regex
L26 QuestionQualityScorer — multi-feature
L27 ReadabilityScorer — textstat
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_jieba_analyse = None
_rapidfuzz = None
_textstat = None


def _get_jieba_analyse():
    global _jieba_analyse
    if _jieba_analyse is None:
        import jieba.analyse
        _jieba_analyse = jieba.analyse
    return _jieba_analyse


def _get_rapidfuzz():
    global _rapidfuzz
    if _rapidfuzz is None:
        from rapidfuzz import fuzz
        _rapidfuzz = fuzz
    return _rapidfuzz


def _get_textstat():
    global _textstat
    if _textstat is None:
        import textstat
        _textstat = textstat
    return _textstat


# ── Domain concept map (expanded from existing topic.py) ──

CONCEPT_MAP = {
    "梯度下降": "梯度下降 (Gradient Descent)", "gradient descent": "梯度下降 (Gradient Descent)",
    "學習率": "學習率 (Learning Rate)", "learning rate": "學習率 (Learning Rate)",
    "損失函數": "損失函數 (Loss Function)", "loss function": "損失函數 (Loss Function)",
    "線性回歸": "線性回歸 (Linear Regression)", "linear regression": "線性回歸 (Linear Regression)",
    "決策邊界": "決策邊界 (Decision Boundary)", "decision boundary": "決策邊界 (Decision Boundary)",
    "邏輯迴歸": "邏輯迴歸 (Logistic Regression)", "logistic regression": "邏輯迴歸 (Logistic Regression)",
    "svm": "支撐向量機 (SVM)", "支撐向量機": "支撐向量機 (SVM)",
    "決策樹": "決策樹 (Decision Tree)", "decision tree": "決策樹 (Decision Tree)",
    "隨機森林": "隨機森林 (Random Forest)", "random forest": "隨機森林 (Random Forest)",
    "shap": "SHAP 值", "過擬合": "過擬合 (Overfitting)", "overfitting": "過擬合 (Overfitting)",
    "欠擬合": "欠擬合 (Underfitting)", "underfitting": "欠擬合 (Underfitting)",
    "交叉驗證": "交叉驗證 (Cross-Validation)", "cross-validation": "交叉驗證 (Cross-Validation)",
    "超參數": "超參數調校 (Hyperparameter Tuning)", "hyperparameter": "超參數調校 (Hyperparameter Tuning)",
    "激活函數": "激活函數 (Activation Function)", "activation function": "激活函數 (Activation Function)",
    "relu": "ReLU 激活函數", "sigmoid": "Sigmoid 函數",
    "神經網路": "神經網路 (Neural Network)", "neural network": "神經網路 (Neural Network)",
    "cnn": "卷積神經網路 (CNN)", "rnn": "循環神經網路 (RNN)",
    "lstm": "LSTM", "transformer": "Transformer",
    "attention": "注意力機制 (Attention)", "注意力": "注意力機制 (Attention)",
    "dropout": "Dropout", "batch normalization": "批次正規化 (BatchNorm)",
    "早停": "早停法 (Early Stopping)", "early stopping": "早停法 (Early Stopping)",
    "特徵工程": "特徵工程 (Feature Engineering)", "feature engineering": "特徵工程 (Feature Engineering)",
    "rag": "檢索增強生成 (RAG)", "embedding": "嵌入 (Embedding)", "嵌入": "嵌入 (Embedding)",
    "mlops": "MLOps", "mlflow": "MLflow",
    "bagging": "Bagging", "boosting": "Boosting", "gbdt": "梯度提升樹 (GBDT)",
    "正則化": "正則化 (Regularization)", "regularization": "正則化 (Regularization)",
}

# ── Known entities (packages, models, algorithms) ──

ENTITY_PACKAGES = {
    "scikit-learn", "sklearn", "numpy", "pandas", "matplotlib", "seaborn",
    "plotly", "pytorch", "tensorflow", "keras", "xgboost", "lightgbm",
    "shap", "mlflow", "huggingface", "transformers",
}

ENTITY_ALGORITHMS = {
    "KNN", "SVM", "GBDT", "XGBoost", "LightGBM", "AdaBoost",
    "ResNet", "VGG", "LeNet", "BERT", "GPT", "GAN", "VAE",
}


# ── L21: Keyword Extractor ──

def keyword_extractor(ctx: NLPContext) -> NLPContext:
    """L21: Extract keywords using jieba TF-IDF + TextRank."""
    analyse = _get_jieba_analyse()

    tfidf_kw = analyse.extract_tags(ctx.user_message, topK=8, withWeight=True)
    textrank_kw = analyse.textrank(ctx.user_message, topK=8, withWeight=True)

    # Merge and deduplicate, keep highest weight
    merged = {}
    for kw, w in tfidf_kw:
        merged[kw] = max(merged.get(kw, 0), w)
    for kw, w in textrank_kw:
        merged[kw] = max(merged.get(kw, 0), w * 0.8)

    sorted_kw = sorted(merged.items(), key=lambda x: -x[1])
    ctx.keywords = [kw for kw, _ in sorted_kw[:10]]
    ctx.keyword_scores = [(kw, round(w, 4)) for kw, w in sorted_kw[:10]]
    return ctx


# ── L22: Domain Concept Matcher ──

def domain_concept_matcher(ctx: NLPContext) -> NLPContext:
    """L22: Match keywords to domain concepts using exact + fuzzy matching."""
    fuzz = _get_rapidfuzz()
    text_lower = ctx.user_message.lower()
    concepts = []
    seen = set()

    # Exact match first
    for trigger, concept in CONCEPT_MAP.items():
        if trigger in text_lower and concept not in seen:
            concepts.append(concept)
            seen.add(concept)

    # Fuzzy match on keywords
    for kw in ctx.keywords[:5]:
        for trigger, concept in CONCEPT_MAP.items():
            if concept in seen:
                continue
            if fuzz.ratio(kw.lower(), trigger) > 80:
                concepts.append(concept)
                seen.add(concept)

    ctx.domain_concepts = concepts
    return ctx


# ── L23: Named Entity Recognizer ──

def named_entity_recognizer(ctx: NLPContext) -> NLPContext:
    """L23: Recognize ML/DL named entities (packages, algorithms, metrics)."""
    text_lower = ctx.user_message.lower()
    entities = []

    for pkg in ENTITY_PACKAGES:
        if pkg.lower() in text_lower:
            entities.append({"text": pkg, "type": "PACKAGE"})

    for algo in ENTITY_ALGORITHMS:
        if algo.lower() in text_lower:
            entities.append({"text": algo, "type": "ALGORITHM"})

    # Detect metric names
    metrics = re.findall(r'\b(accuracy|precision|recall|f1|auc|rmse|mae|mse|r2)\b', text_lower)
    for m in metrics:
        entities.append({"text": m.upper(), "type": "METRIC"})

    ctx.named_entities = entities
    return ctx


# ── L24: Code Block Detector ──

CODE_PATTERNS = [
    r'```[\s\S]*?```',
    r'^\s*(import |from .+ import |def |class |print\(|for .+ in |if __name__)',
    r'^\s*\w+\s*=\s*\w+\.\w+\(',
    r'model\.(fit|predict|transform|score)\(',
    r'plt\.\w+\(',
    r'pd\.(read_csv|DataFrame)',
    r'np\.\w+\(',
]


def code_block_detector(ctx: NLPContext) -> NLPContext:
    """L24: Detect code blocks and identify programming language."""
    text = ctx.user_message

    # Fenced code blocks
    fenced = re.findall(r'```(\w*)\n?([\s\S]*?)```', text)
    if fenced:
        ctx.has_code = True
        ctx.code_blocks = [code for _, code in fenced]
        ctx.code_language = fenced[0][0] or "python"
        return ctx

    # Inline code patterns
    for pattern in CODE_PATTERNS[1:]:
        if re.search(pattern, text, re.MULTILINE):
            ctx.has_code = True
            ctx.code_language = "python"
            return ctx

    ctx.has_code = False
    return ctx


# ── L25: Math Expression Detector ──

MATH_PATTERNS = [
    r'\$\$.+?\$\$',
    r'\$.+?\$',
    r'\\frac\{', r'\\sum', r'\\nabla', r'\\partial',
    r'\u2211|\u220f|\u222b|\u2202|\u2207|\u2208|\u2209|\u2282|\u2283|\u2200|\u2203',
    r'\b[a-z]\s*=\s*[a-z]\s*[\+\-\*/]\s*[a-z]\b',
    r'argmax|argmin|log\s*\(|exp\s*\(',
]


def math_expression_detector(ctx: NLPContext) -> NLPContext:
    """L25: Detect mathematical expressions in the message."""
    text = ctx.user_message
    expressions = []

    # LaTeX blocks
    for expr in re.findall(r'\$\$(.+?)\$\$', text, re.DOTALL):
        expressions.append(expr.strip())
    for expr in re.findall(r'\$(.+?)\$', text):
        expressions.append(expr.strip())

    if expressions:
        ctx.has_math = True
        ctx.math_expressions = expressions
        return ctx

    # Other math indicators
    for pattern in MATH_PATTERNS[4:]:
        if re.search(pattern, text):
            ctx.has_math = True
            return ctx

    ctx.has_math = False
    return ctx


# ── L26: Question Quality Scorer ──

def question_quality_scorer(ctx: NLPContext) -> NLPContext:
    """L26: Score question quality (0-1) to guide students to ask better questions."""
    score = 0.3  # Base score
    feedback = []

    # Has specific context
    if ctx.keywords and len(ctx.keywords) >= 2:
        score += 0.1
    else:
        feedback.append("可以加入更具體的關鍵詞")

    # References week or topic
    if re.search(r'第\s*\d+\s*週|week\s*\d+', ctx.user_message, re.IGNORECASE):
        score += 0.1

    # Shows prior attempt
    if any(w in ctx.user_message.lower() for w in ["我試了", "我嘗試", "i tried", "my attempt"]):
        score += 0.15
    else:
        feedback.append("描述你已經嘗試過什麼會更有幫助")

    # Includes error message or output
    if ctx.has_code or any(w in ctx.user_message.lower() for w in ["error", "錯誤", "output"]):
        score += 0.1

    # Appropriate length (not too short)
    if len(ctx.user_message) > 30:
        score += 0.1
    else:
        feedback.append("問題可以再描述得更詳細一些")

    # Domain concepts identified
    if ctx.domain_concepts:
        score += 0.1

    ctx.question_quality = min(score, 1.0)
    ctx.quality_feedback = "\uff1b".join(feedback) if feedback else ""
    return ctx


# ── L27: Readability Scorer ──

def readability_scorer(ctx: NLPContext) -> NLPContext:
    """L27: Score text readability/complexity using textstat."""
    ts = _get_textstat()
    try:
        # textstat works best on English; for Chinese, use character count heuristic
        if ctx.language == "en":
            ctx.readability_score = ts.flesch_reading_ease(ctx.user_message) / 100.0
        else:
            # Simple heuristic for Chinese: sentence length and term complexity
            avg_sent_len = len(ctx.user_message) / max(len(ctx.sentences), 1)
            ctx.readability_score = min(avg_sent_len / 50.0, 1.0)
    except Exception:
        ctx.readability_score = 0.5
    return ctx


# ── Public aliases (used by __init__.py FULL_PIPELINE) ──

recognize_entities = named_entity_recognizer
detect_code_blocks = code_block_detector
detect_math = math_expression_detector
score_question_quality = question_quality_scorer
score_readability = readability_scorer

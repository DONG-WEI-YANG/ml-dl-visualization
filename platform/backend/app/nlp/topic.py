"""Layer 4: Topic Extraction — TF-IDF keyword scoring with domain concept mapping."""

import re
import logging
from .pipeline import NLPContext
from .trainer import load_model

logger = logging.getLogger(__name__)

# ── Cached corpus model ──
_corpus_data = None
_corpus_loaded = False


def _get_corpus():
    """Lazy-load the corpus TF-IDF data."""
    global _corpus_data, _corpus_loaded
    if not _corpus_loaded:
        _corpus_data = load_model("corpus_tfidf")
        _corpus_loaded = True
        if _corpus_data:
            logger.info("Corpus TF-IDF loaded (%d chunks)", len(_corpus_data.get("ids", [])))
        else:
            logger.info("No corpus TF-IDF found — using basic keyword extraction")
    return _corpus_data


def reload_model():
    """Force reload after retraining."""
    global _corpus_loaded
    _corpus_loaded = False


STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "這",
    "上", "也", "到", "說", "要", "會", "可以", "請問", "想", "個", "中", "嗎",
    "怎麼", "什麼", "為什麼", "如何", "能", "用", "讓", "吧", "呢", "跟", "從",
    "那", "被", "把", "給", "很", "太", "再", "還", "去", "來", "做", "看",
    "但", "所以", "因為", "如果", "然後", "或", "而", "對", "比", "以", "其",
    "the", "is", "a", "an", "in", "of", "to", "and", "for", "it", "this", "that",
    "how", "what", "why", "can", "do", "i", "my", "me", "be", "are", "was",
    "with", "on", "at", "by", "from", "or", "but", "not", "so", "if", "then",
}

CONCEPT_MAP = {
    "梯度下降": "梯度下降 (Gradient Descent)",
    "gradient descent": "梯度下降 (Gradient Descent)",
    "學習率": "學習率 (Learning Rate)",
    "learning rate": "學習率 (Learning Rate)",
    "損失函數": "損失函數 (Loss Function)",
    "loss function": "損失函數 (Loss Function)",
    "線性回歸": "線性回歸 (Linear Regression)",
    "linear regression": "線性回歸 (Linear Regression)",
    "決策邊界": "決策邊界 (Decision Boundary)",
    "decision boundary": "決策邊界 (Decision Boundary)",
    "邏輯迴歸": "邏輯迴歸 (Logistic Regression)",
    "logistic regression": "邏輯迴歸 (Logistic Regression)",
    "svm": "支撐向量機 (SVM)",
    "支撐向量機": "支撐向量機 (SVM)",
    "核方法": "核方法 (Kernel Methods)",
    "kernel": "核方法 (Kernel Methods)",
    "決策樹": "決策樹 (Decision Tree)",
    "decision tree": "決策樹 (Decision Tree)",
    "隨機森林": "隨機森林 (Random Forest)",
    "random forest": "隨機森林 (Random Forest)",
    "特徵重要度": "特徵重要度 (Feature Importance)",
    "feature importance": "特徵重要度 (Feature Importance)",
    "shap": "SHAP 值",
    "過擬合": "過擬合 (Overfitting)",
    "overfitting": "過擬合 (Overfitting)",
    "欠擬合": "欠擬合 (Underfitting)",
    "underfitting": "欠擬合 (Underfitting)",
    "交叉驗證": "交叉驗證 (Cross-Validation)",
    "cross-validation": "交叉驗證 (Cross-Validation)",
    "超參數": "超參數調校 (Hyperparameter Tuning)",
    "hyperparameter": "超參數調校 (Hyperparameter Tuning)",
    "激活函數": "激活函數 (Activation Function)",
    "activation function": "激活函數 (Activation Function)",
    "relu": "ReLU 激活函數",
    "sigmoid": "Sigmoid 函數",
    "神經網路": "神經網路 (Neural Network)",
    "neural network": "神經網路 (Neural Network)",
    "卷積": "卷積神經網路 (CNN)",
    "cnn": "卷積神經網路 (CNN)",
    "rnn": "循環神經網路 (RNN)",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "attention": "注意力機制 (Attention)",
    "注意力": "注意力機制 (Attention)",
    "正則化": "正則化 (Regularization)",
    "regularization": "正則化 (Regularization)",
    "dropout": "Dropout",
    "batch normalization": "批次正規化 (Batch Normalization)",
    "早停": "早停法 (Early Stopping)",
    "early stopping": "早停法 (Early Stopping)",
    "特徵工程": "特徵工程 (Feature Engineering)",
    "feature engineering": "特徵工程 (Feature Engineering)",
    "標準化": "標準化 (Standardization)",
    "正規化": "正規化 (Normalization)",
}


def _extract_keywords_basic(text: str) -> list[str]:
    """Basic keyword extraction using stop-word splitting."""
    keywords = []

    # English words (3+ chars)
    eng_words = re.findall(r"[a-zA-Z]{3,}", text)
    for w in eng_words:
        if w.lower() not in STOP_WORDS:
            keywords.append(w)

    # Chinese: split by stop words, keep meaningful segments
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    for segment in chinese:
        clean = segment
        for sw in sorted(STOP_WORDS, key=len, reverse=True):
            clean = clean.replace(sw, "|")
        parts = [p for p in clean.split("|") if len(p) >= 2]
        if parts:
            keywords.extend(parts)
        elif len(segment) >= 2:
            keywords.append(segment)

    # Deduplicate
    seen = set()
    unique = []
    for k in keywords:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            unique.append(k)
    return unique[:10]


def _score_keywords_tfidf(text: str, keywords: list[str]) -> list[str]:
    """Re-rank keywords by TF-IDF importance from corpus vocabulary."""
    corpus = _get_corpus()
    if corpus is None:
        return keywords

    vectorizer = corpus["vectorizer"]
    vocab = vectorizer.vocabulary_

    # Score each keyword by how many corpus features it matches
    scored = []
    for kw in keywords:
        # Check if keyword's character n-grams appear in corpus vocabulary
        kw_lower = kw.lower()
        score = 0
        for n in range(2, min(len(kw_lower) + 1, 5)):
            for i in range(len(kw_lower) - n + 1):
                ngram = kw_lower[i:i + n]
                if ngram in vocab:
                    score += 1
        scored.append((kw, score))

    # Sort by TF-IDF relevance score (descending), keep original order for ties
    scored.sort(key=lambda x: -x[1])
    return [kw for kw, _ in scored]


def extract_topics(ctx: NLPContext) -> NLPContext:
    """Extract keywords and map to domain concepts — TF-IDF enhanced."""
    text = ctx.user_message

    # Step 1: Basic keyword extraction
    keywords = _extract_keywords_basic(text)

    # Step 2: TF-IDF re-ranking (if corpus model available)
    keywords = _score_keywords_tfidf(text, keywords)
    ctx.keywords = keywords[:10]

    # Step 3: Map to domain concepts (always rule-based — high precision needed)
    text_lower = text.lower()
    concepts = []
    seen_concepts = set()
    for trigger, concept in CONCEPT_MAP.items():
        if trigger in text_lower and concept not in seen_concepts:
            concepts.append(concept)
            seen_concepts.add(concept)
    ctx.domain_concepts = concepts

    return ctx

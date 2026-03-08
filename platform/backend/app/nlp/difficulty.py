"""Layer 3: Difficulty Assessment — multi-feature assessment with jieba + wordfreq.

Estimates student level from language cues using:
1. jieba token analysis (technical term density)
2. Message length and complexity
3. POS tag distribution from ctx.pos_tags
4. wordfreq-based vocabulary rarity
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

# Technical terms indicate higher proficiency
ADVANCED_TERMS = {
    # ML
    "overfitting", "underfitting", "regularization", "cross-validation",
    "gradient descent", "backpropagation", "loss function", "cost function",
    "hyperparameter", "learning rate", "batch size", "epoch",
    "bias-variance", "ensemble", "bagging", "boosting",
    # DL
    "convolution", "pooling", "activation function", "dropout",
    "batch normalization", "attention", "transformer", "embedding",
    "autoencoder", "GAN", "transfer learning",
    # Chinese
    "過擬合", "欠擬合", "正則化", "交叉驗證", "梯度下降",
    "反向傳播", "損失函數", "超參數", "學習率", "批次大小",
    "偏差-方差", "集成學習", "卷積", "池化", "激活函數",
    "注意力機制", "嵌入", "遷移學習", "特徵工程",
}

BEGINNER_SIGNALS = [
    r"初學", r"新手", r"剛開始", r"第一次", r"入門", r"基礎",
    r"不知道從哪", r"完全不會", r"零基礎",
    r"beginner", r"newbie", r"just started", r"first time",
]

ADVANCED_SIGNALS = [
    r"論文", r"paper", r"研究", r"research", r"最佳化", r"optimization",
    r"收斂", r"convergence", r"數學推導", r"理論",
    r"state.of.the.art", r"SOTA", r"ablation",
]


def assess_difficulty(ctx: NLPContext) -> NLPContext:
    """Assess student difficulty level using multi-feature analysis."""
    text = ctx.user_message
    text_lower = text.lower()

    # Feature 1: Count technical terms used (original)
    tech_count = sum(1 for term in ADVANCED_TERMS if term.lower() in text_lower)
    ctx.uses_technical_terms = tech_count > 0

    # Feature 2: Check explicit level signals
    is_beginner = any(re.search(p, text_lower) for p in BEGINNER_SIGNALS)
    is_advanced = any(re.search(p, text_lower) for p in ADVANCED_SIGNALS)

    # Feature 3: jieba token analysis for technical term density
    jieba_tech_density = 0.0
    try:
        import jieba
        tokens = list(jieba.cut(text))
        meaningful_tokens = [t for t in tokens if len(t.strip()) > 1]
        if meaningful_tokens:
            tech_in_tokens = sum(
                1 for t in meaningful_tokens
                if t.lower() in {term.lower() for term in ADVANCED_TERMS}
            )
            jieba_tech_density = tech_in_tokens / len(meaningful_tokens)
    except ImportError:
        pass

    # Feature 4: POS tag distribution (if available)
    pos_tech_ratio = 0.0
    if ctx.pos_tags:
        tech_pos = {"eng", "n", "nz", "nr"}
        tech_pos_count = sum(1 for _, flag in ctx.pos_tags if flag in tech_pos)
        total_pos = len(ctx.pos_tags)
        if total_pos > 0:
            pos_tech_ratio = tech_pos_count / total_pos

    # Feature 5: wordfreq-based vocabulary rarity
    vocab_rarity = 0.0
    try:
        from wordfreq import word_frequency
        import jieba
        tokens = list(jieba.cut(text))
        rarity_scores = []
        for t in tokens:
            if len(t.strip()) < 2:
                continue
            freq = max(word_frequency(t, 'zh'), word_frequency(t, 'en'))
            if freq == 0:
                rarity_scores.append(1.0)
            elif freq < 1e-5:
                rarity_scores.append(0.7)
            elif freq < 1e-4:
                rarity_scores.append(0.4)
            else:
                rarity_scores.append(0.1)
        if rarity_scores:
            vocab_rarity = sum(rarity_scores) / len(rarity_scores)
    except ImportError:
        pass

    # Combine features for level assessment
    # Weighted score: higher = more advanced
    combined_score = (
        tech_count * 0.15 +          # Each technical term adds weight
        jieba_tech_density * 2.0 +    # Technical density
        pos_tech_ratio * 1.5 +        # POS ratio
        vocab_rarity * 1.0            # Vocabulary rarity
    )

    # Apply explicit signals
    if is_advanced or combined_score >= 1.5 or tech_count >= 3:
        ctx.student_level = "advanced"
    elif is_beginner or (tech_count == 0 and combined_score < 0.5):
        ctx.student_level = "beginner"
    else:
        ctx.student_level = "intermediate"

    # Assess question complexity
    msg_len = len(text)
    has_multiple_questions = text.count("？") + text.count("?") > 1
    has_code = bool(re.search(r"```|def |import |class |print\(", text))

    if has_multiple_questions or has_code or msg_len > 200:
        ctx.question_complexity = "complex"
    elif msg_len > 80 or tech_count >= 2:
        ctx.question_complexity = "moderate"
    else:
        ctx.question_complexity = "simple"

    return ctx

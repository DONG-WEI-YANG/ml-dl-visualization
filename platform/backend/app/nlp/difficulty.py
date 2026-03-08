"""Layer 3: Difficulty Assessment — estimate student's level from their language."""

import re
from .pipeline import NLPContext

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
    """Assess student difficulty level from language cues."""
    text_lower = ctx.user_message.lower()

    # Count technical terms used
    tech_count = sum(1 for term in ADVANCED_TERMS if term.lower() in text_lower)
    ctx.uses_technical_terms = tech_count > 0

    # Check explicit level signals
    is_beginner = any(re.search(p, text_lower) for p in BEGINNER_SIGNALS)
    is_advanced = any(re.search(p, text_lower) for p in ADVANCED_SIGNALS)

    # Assess level
    if is_advanced or tech_count >= 3:
        ctx.student_level = "advanced"
    elif is_beginner or tech_count == 0:
        ctx.student_level = "beginner"
    else:
        ctx.student_level = "intermediate"

    # Assess question complexity
    msg_len = len(ctx.user_message)
    has_multiple_questions = ctx.user_message.count("？") + ctx.user_message.count("?") > 1
    has_code = bool(re.search(r"```|def |import |class |print\(", ctx.user_message))

    if has_multiple_questions or has_code or msg_len > 200:
        ctx.question_complexity = "complex"
    elif msg_len > 80 or tech_count >= 2:
        ctx.question_complexity = "moderate"
    else:
        ctx.question_complexity = "simple"

    return ctx

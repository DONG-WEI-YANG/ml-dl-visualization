"""Coordinator layers — aggregate, resolve conflicts, adaptive routing."""

import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("numpy not available for coordinators")


# ── C1: Aggregate Confidence ──

def aggregate_confidence(ctx: NLPContext) -> NLPContext:
    """C1: Collect confidence scores from multiple layers and compute weighted average.

    Placed after L14 (after all understanding layers).
    """
    scores = {}
    weights = {}

    # Collect available confidence scores
    if ctx.intent_confidence > 0:
        scores["intent"] = ctx.intent_confidence
        weights["intent"] = 0.3
    if ctx.sub_intent_confidence > 0:
        scores["sub_intent"] = ctx.sub_intent_confidence
        weights["sub_intent"] = 0.15
    if ctx.emotion_confidence > 0:
        scores["emotion"] = ctx.emotion_confidence
        weights["emotion"] = 0.15
    # Sentiment is 0-1 where 0.5 is neutral; convert to confidence-like score
    sentiment_conf = abs(ctx.sentiment_score - 0.5) * 2  # 0 at neutral, 1 at extremes
    scores["sentiment"] = sentiment_conf
    weights["sentiment"] = 0.1
    if ctx.question_quality > 0:
        scores["question_quality"] = ctx.question_quality
        weights["question_quality"] = 0.2
    # Politeness as a signal (higher = more engaged)
    scores["politeness"] = ctx.politeness_score
    weights["politeness"] = 0.1

    # Compute weighted average
    if scores and np is not None:
        total_weight = sum(weights.get(k, 0.1) for k in scores)
        if total_weight > 0:
            weighted_sum = sum(scores[k] * weights.get(k, 0.1) for k in scores)
            overall_confidence = weighted_sum / total_weight
        else:
            overall_confidence = 0.5
    else:
        overall_confidence = 0.5

    # Store as metadata (reuse confidence_level if it would be informative)
    # Don't overwrite confidence_level as it represents student confidence
    # Instead, we store this in layers_executed metadata
    ctx.layers_executed.append(f"aggregate_confidence={overall_confidence:.2f}")

    # If overall confidence is very low, note it for response adjustments
    if overall_confidence < 0.3:
        if not ctx.quality_feedback:
            ctx.quality_feedback = "系統對此回答的信心較低，建議更具體地描述你的問題"
        elif "信心較低" not in ctx.quality_feedback:
            ctx.quality_feedback += "；系統對此回答的信心較低，建議更具體地描述你的問題"

    return ctx


# ── C2: Resolve Conflicts ──

def resolve_conflicts(ctx: NLPContext) -> NLPContext:
    """C2: Check for and resolve contradictions between layer outputs.

    Placed after L27 (after all analysis layers).
    """
    resolutions = []

    # Conflict 1: emotion=frustrated but sentiment_score > 0.7
    if ctx.emotion == "frustrated" and ctx.sentiment_score > 0.7:
        # Trust emotion (regex/ML is more specific than general sentiment)
        resolutions.append("emotion:frustrated+sentiment:high→trust_emotion")
        # No change: keep emotion as-is, regex pattern match is more reliable

    # Conflict 2: intent=definition but has_code=True
    if ctx.intent == "definition" and ctx.has_code:
        # Likely a code question, not definition
        ctx.intent = "code"
        ctx.sub_intent = "code_review"
        resolutions.append("intent:definition+has_code→intent:code")

    # Conflict 3: student_level=beginner but vocabulary_score > 0.7
    if ctx.student_level == "beginner" and ctx.vocabulary_score > 0.7:
        ctx.student_level = "intermediate"
        resolutions.append("level:beginner+vocab:high→level:intermediate")

    # Conflict 4: emotion=confident but many confusion signals in text
    if ctx.emotion == "confident" and ctx.frustration_level >= 3:
        ctx.emotion = "confused"
        resolutions.append("emotion:confident+frustration:high→emotion:confused")

    # Conflict 5: high urgency but low politeness might indicate stress
    if ctx.urgency == "high" and ctx.politeness_score < 0.3:
        # Student might be stressed, not rude — adjust frustration up
        ctx.frustration_level = min(ctx.frustration_level + 1, 5)
        resolutions.append("urgency:high+politeness:low→frustration+1")

    # Conflict 6: learning_style=textual but intent=visualization
    if ctx.learning_style == "textual" and ctx.intent == "visualization":
        ctx.learning_style = "visual"
        resolutions.append("style:textual+intent:viz→style:visual")

    if resolutions:
        ctx.layers_executed.append(f"conflicts_resolved={','.join(resolutions)}")

    return ctx


# ── C3: Route Pipeline ──

def route_pipeline(ctx: NLPContext) -> NLPContext:
    """C3: Determine routing metadata for the pipeline.

    Placed at the very beginning (L0 position, before preprocessing).
    This is informational — actual skipping would need pipeline architecture changes.
    """
    routing = []

    # For very short messages (<5 chars), note that many layers may be unreliable
    if len(ctx.user_message.strip()) < 5:
        routing.append("short_message:analysis_limited")

    # For pure English messages, note Chinese-specific layers had limited effect
    # We check for Chinese characters
    has_chinese = bool(
        __import__("re").search(r'[\u4e00-\u9fff]', ctx.user_message)
    )
    if not has_chinese:
        routing.append("pure_english:zh_layers_limited")

    # For very long messages (>500 chars), note extra analysis may be needed
    if len(ctx.user_message) > 500:
        routing.append("long_message:full_analysis")

    # For messages with code blocks, note code-specific layers are important
    if "```" in ctx.user_message or "import " in ctx.user_message:
        routing.append("has_code:code_analysis_priority")

    # Store routing decisions in metadata
    if routing:
        ctx.layers_executed.append(f"routing={','.join(routing)}")

    return ctx


# ── Public aliases ──

aggregate = aggregate_confidence
resolve = resolve_conflicts
route = route_pipeline

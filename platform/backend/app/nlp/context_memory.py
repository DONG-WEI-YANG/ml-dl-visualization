"""Group E: Context & Memory Layers (L28-32).

L28 ConversationTracker — enhanced from existing
L29 TopicContinuityDetector — sentence-transformers
L30 HintLadderManager — state machine
L31 SessionSummarizer — snownlp + jieba
L32 KnowledgeStateTracker — concept tracking
"""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

_sentence_model = None


def _get_sentence_model():
    """Lazy-load sentence-transformers model."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")
            logger.info("Sentence-transformers model loaded")
        except Exception as e:
            logger.warning("Could not load sentence-transformers: %s", e)
            _sentence_model = False  # Mark as failed
    return _sentence_model if _sentence_model is not False else None


# ── L28: Conversation Tracker (enhanced) ──

FOLLOWUP_SIGNALS = [
    r"^那", r"^還有", r"^另外", r"^接著", r"^然後呢", r"^繼續",
    r"^所以", r"^也就是", r"^換句話說", r"^你剛才說",
    r"^上面", r"^前面", r"^剛剛", r"^這個",
    r"^then", r"^also", r"^and ", r"^what about", r"^so ",
    r"^following up", r"^you (said|mentioned)",
]


def conversation_tracker(ctx: NLPContext) -> NLPContext:
    """L28: Track multi-turn conversation state."""
    history = ctx.conversation_history
    ctx.turn_count = sum(1 for m in history if m.get("role") == "user")

    # Detect follow-up
    text = ctx.user_message
    ctx.is_followup = ctx.turn_count > 0 and any(
        re.search(p, text, re.IGNORECASE) for p in FOLLOWUP_SIGNALS
    )

    if ctx.turn_count > 0:
        ctx.previous_intent = ctx.intent

    return ctx


# ── L29: Topic Continuity Detector ──

def topic_continuity_detector(ctx: NLPContext) -> NLPContext:
    """L29: Detect if current question continues the previous topic using embeddings."""
    if ctx.turn_count == 0:
        ctx.topic_continuity = 0.0
        return ctx

    model = _get_sentence_model()
    if model is None:
        # Fallback: simple keyword overlap
        prev_messages = [m["content"] for m in ctx.conversation_history if m.get("role") == "user"]
        if prev_messages:
            prev_words = set(prev_messages[-1].lower().split())
            curr_words = set(ctx.user_message.lower().split())
            overlap = len(prev_words & curr_words)
            ctx.topic_continuity = min(overlap / max(len(curr_words), 1), 1.0)
        return ctx

    try:
        prev_messages = [m["content"] for m in ctx.conversation_history if m.get("role") == "user"]
        if prev_messages:
            embeddings = model.encode([prev_messages[-1], ctx.user_message])
            from sklearn.metrics.pairwise import cosine_similarity
            sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0])
            ctx.topic_continuity = max(sim, 0.0)
            if sim > 0.6:
                ctx.continued_topic = ctx.topic
    except Exception:
        ctx.topic_continuity = 0.0

    return ctx


# ── L30: Hint Ladder Manager ──

PROGRESS_SIGNALS = [
    r"我試了", r"我嘗試", r"我寫了", r"結果是", r"我得到",
    r"但是", r"不過", r"可是", r"出現了", r"變成",
    r"i tried", r"i got", r"my result", r"but then", r"however",
]

STUCK_SIGNALS = [
    r"還是不懂", r"還是不行", r"一樣的錯", r"又失敗",
    r"看不出", r"不知道哪裡", r"完全沒有頭緒",
    r"still", r"again", r"same error", r"doesn't work",
]


def hint_ladder_manager(ctx: NLPContext) -> NLPContext:
    """L30: Manage Hint Ladder progression (1-4)."""
    text = ctx.user_message
    has_progress = any(re.search(p, text, re.IGNORECASE) for p in PROGRESS_SIGNALS)
    still_stuck = any(re.search(p, text, re.IGNORECASE) for p in STUCK_SIGNALS)

    if ctx.turn_count == 0:
        ctx.hint_level = 1
    elif still_stuck and ctx.turn_count >= 3:
        ctx.hint_level = 4
    elif still_stuck:
        ctx.hint_level = min(ctx.hint_level + 1, 4)
    elif has_progress:
        ctx.hint_level = min(ctx.hint_level + 1, 3)
    elif ctx.turn_count >= 2:
        ctx.hint_level = min(ctx.turn_count, 3)
    else:
        ctx.hint_level = 1

    # Frustration-based override
    if ctx.frustration_level >= 4:
        ctx.hint_level = max(ctx.hint_level, 3)

    return ctx


# ── L31: Session Summarizer ──

def session_summarizer(ctx: NLPContext) -> NLPContext:
    """L31: Generate a brief summary of the conversation so far."""
    if ctx.turn_count < 2:
        ctx.session_summary = ""
        return ctx

    user_msgs = [m["content"] for m in ctx.conversation_history if m.get("role") == "user"]
    if not user_msgs:
        return ctx

    try:
        from snownlp import SnowNLP
        combined = "\u3002".join(user_msgs[-3:])  # Last 3 messages
        s = SnowNLP(combined)
        summaries = s.summary(3)
        ctx.session_summary = "\uff1b".join(summaries) if summaries else ""
    except Exception:
        # Fallback: just use keywords
        ctx.session_summary = "\u3001".join(ctx.keywords[:5]) if ctx.keywords else ""

    return ctx


# ── L32: Knowledge State Tracker ──

def knowledge_state_tracker(ctx: NLPContext) -> NLPContext:
    """L32: Track which concepts the student has demonstrated understanding of."""
    # Analyze conversation history for confirmed understanding
    for msg in ctx.conversation_history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "").lower()
        # Signals of understanding
        if any(w in content for w in ["我懂了", "了解了", "原來", "i see", "i understand", "got it", "makes sense"]):
            # Find which concepts were in the previous assistant message
            idx = ctx.conversation_history.index(msg)
            if idx > 0:
                prev = ctx.conversation_history[idx - 1].get("content", "").lower()
                for concept in ctx.domain_concepts:
                    if concept.split("(")[0].strip().lower() in prev:
                        if concept not in ctx.known_concepts:
                            ctx.known_concepts.append(concept)

    return ctx


# ── Public aliases (used by __init__.py FULL_PIPELINE) ──

detect_topic_continuity = topic_continuity_detector
summarize_session = session_summarizer
track_knowledge_state = knowledge_state_tracker

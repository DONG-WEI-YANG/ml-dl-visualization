"""Layer 5: Conversation Tracker — multi-turn context and Hint Ladder state."""

import re
from .pipeline import NLPContext

FOLLOWUP_SIGNALS = [
    r"^那", r"^還有", r"^另外", r"^接著", r"^然後呢", r"^繼續",
    r"^所以", r"^也就是", r"^換句話說", r"^你剛才說",
    r"^上面", r"^前面", r"^剛剛", r"^這個",
    r"^then", r"^also", r"^and ", r"^what about", r"^so ",
    r"^following up", r"^you (said|mentioned)",
]

# Hint Ladder: detect if student is showing progress
PROGRESS_SIGNALS = [
    r"我試了", r"我嘗試", r"我寫了", r"結果是", r"我得到",
    r"但是", r"不過", r"可是", r"出現了", r"變成",
    r"i tried", r"i got", r"my result", r"but then", r"however",
]

STILL_STUCK_SIGNALS = [
    r"還是不懂", r"還是不行", r"一樣的錯", r"又失敗",
    r"看不出", r"不知道哪裡", r"完全沒有頭緒",
    r"still", r"again", r"same error", r"doesn't work",
]


def track_conversation(ctx: NLPContext) -> NLPContext:
    """Track multi-turn conversation state and Hint Ladder level."""
    history = ctx.conversation_history
    ctx.turn_count = sum(1 for m in history if m.get("role") == "user")

    # Detect follow-up
    text = ctx.user_message
    ctx.is_followup = ctx.turn_count > 0 and any(
        re.search(p, text, re.IGNORECASE) for p in FOLLOWUP_SIGNALS
    )

    # Track previous intent (from last assistant response context)
    if ctx.turn_count > 0:
        ctx.previous_intent = ctx.intent  # Will be the current one; useful for next turn

    # Hint Ladder progression
    has_progress = any(re.search(p, text, re.IGNORECASE) for p in PROGRESS_SIGNALS)
    still_stuck = any(re.search(p, text, re.IGNORECASE) for p in STILL_STUCK_SIGNALS)

    if ctx.turn_count == 0:
        ctx.hint_level = 1  # Start with clarification
    elif still_stuck and ctx.turn_count >= 3:
        ctx.hint_level = 4  # Give partial example after multiple stuck attempts
    elif still_stuck:
        ctx.hint_level = min(ctx.hint_level + 1, 4)
    elif has_progress:
        ctx.hint_level = min(ctx.hint_level + 1, 3)  # Progress → next level, max step guidance
    elif ctx.turn_count >= 2:
        ctx.hint_level = min(ctx.turn_count, 3)  # Gradually escalate
    else:
        ctx.hint_level = 1

    return ctx

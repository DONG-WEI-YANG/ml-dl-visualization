"""Fast, conservative course-scoped draft answers.

This module intentionally avoids importing the full NLP pipeline.
"""

import re

from app.rag.retriever import retrieve_context

_HIGH_RISK = re.compile(
    r"胸痛|吃什麼藥|劑量|診斷|自殺|自傷|法律意見|投資建議|保證獲利",
    re.IGNORECASE,
)


def _first_sentences(text: str, limit: int = 2) -> list[str]:
    cleaned = re.sub(r"\[[^\]]+\]\s*", "", text.replace("\n", " "))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    parts = [part.strip() for part in re.split(r"(?<=[。！？])", cleaned) if part.strip()]
    return parts[:limit]


def build_quick_answer(question: str, week: int, topic: str) -> str | None:
    """Return a short draft using only current-week retrieval and safe framing."""
    if _HIGH_RISK.search(question):
        return "這個問題涉及高風險的專業判斷，我不能提供醫療判斷或直接處置建議。請立即諮詢合格的專業人員；AI 助教會把回答限制在本課程的 ML/DL 概念。"

    context = retrieve_context(question, week=week, top_k=2)[:1200]
    points = _first_sentences(context, limit=2)
    if not points:
        return f"這題屬於第 {week} 週「{topic}」的範圍。先抓住問題中的核心名詞與它在模型流程中的作用；我正在核對教材，稍後會補上完整說明。"

    core = "".join(points)
    answer = f"先說核心：{core}這與第 {week} 週「{topic}」直接相關。我正在核對完整教材，稍後會提供經驗證的詳細回答。"
    return answer[:499].rstrip("，；： ") + ("。" if not answer[:499].endswith(("。", "！", "？")) else "")

from .base import LLMProvider, LLMMessage
from .prompts import SYSTEM_TUTOR, SYSTEM_HOMEWORK, SYSTEM_TUTOR_NO_RAG
from app.rag.retriever import retrieve_context


def _get_student_context(student_id: str | None) -> str:
    """Build personalized context from student learning records."""
    if not student_id:
        return ""
    try:
        from app.analytics.tracker import get_student_analytics
        analytics = get_student_analytics(student_id)
        if not analytics:
            return ""

        parts = []
        if analytics.get("total_weeks_completed"):
            parts.append(f"已完成 {analytics['total_weeks_completed']} 週")
        if analytics.get("average_score"):
            parts.append(f"平均分數 {analytics['average_score']:.0f}")
        if analytics.get("llm_topics"):
            topics = [t["topic"] for t in analytics["llm_topics"][:5]]
            parts.append(f"常問主題: {', '.join(topics)}")

        if parts:
            return "\n\n【學生學習歷程】" + " | ".join(parts) + "\n請根據此學生的程度調整回覆深度。"
        return ""
    except Exception:
        return ""


class AITutor:
    def __init__(self, provider: LLMProvider, use_rag: bool = True):
        self.provider = provider
        self.use_rag = use_rag

    def _build_system(
        self, messages: list[LLMMessage], week: int, topic: str,
        mode: str, student_id: str | None = None,
    ) -> str:
        # Extract the latest user message for retrieval
        user_query = ""
        for m in reversed(messages):
            if m.role == "user":
                user_query = m.content
                break

        # Personalization from learning records
        student_ctx = _get_student_context(student_id)

        if self.use_rag and user_query:
            rag_context = retrieve_context(user_query, week=week)
            if rag_context:
                if mode == "homework":
                    return SYSTEM_HOMEWORK.format(assignment=topic, rag_context=rag_context) + student_ctx
                return SYSTEM_TUTOR.format(week=week, topic=topic, rag_context=rag_context) + student_ctx

        # Fallback: no RAG context available
        if mode == "homework":
            return SYSTEM_HOMEWORK.format(assignment=topic, rag_context="（無相關教材）") + student_ctx
        return SYSTEM_TUTOR_NO_RAG.format(week=week, topic=topic) + student_ctx

    async def ask(
        self, messages: list[LLMMessage], week: int, topic: str, mode: str = "tutor"
    ):
        system = self._build_system(messages, week, topic, mode)
        return await self.provider.chat(messages, system=system)

    async def ask_stream(
        self, messages: list[LLMMessage], week: int, topic: str, mode: str = "tutor"
    ):
        system = self._build_system(messages, week, topic, mode)
        async for chunk in self.provider.stream(messages, system=system):
            yield chunk

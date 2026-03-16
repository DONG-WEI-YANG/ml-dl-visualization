"""Local NLP provider — no external API needed.

Uses the FULL 42-layer NLP pipeline for deep question analysis and response generation.
Works offline, zero cost, always available.
"""

import re
import logging
from .base import LLMProvider, LLMMessage, LLMResponse
from app.nlp.pipeline import NLPContext, run_pipeline
from app.nlp import FULL_PIPELINE

logger = logging.getLogger(__name__)


class LocalProvider(LLMProvider):
    """Local NLP provider using full 42-layer pipeline. No API needed."""

    def __init__(self):
        self.model_name = "local-nlp-v3"

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        response_text = self._generate(messages, system)
        return LLMResponse(content=response_text, model=self.model_name)

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        """Simulate streaming by yielding chunks of the full response."""
        response_text = self._generate(messages, system)
        # Split at sentence boundaries for natural streaming
        chunks = re.split(r"(?<=[。！？\n])", response_text)
        for chunk in chunks:
            if chunk:
                yield chunk

    def _generate(self, messages: list[LLMMessage], system: str) -> str:
        # Get latest user message
        user_msg = ""
        for m in reversed(messages):
            if m.role == "user":
                user_msg = m.content
                break

        if not user_msg:
            return "請輸入你的問題，我會根據課程教材幫你找到答案。"

        # Extract week from system prompt
        week = 1
        week_match = re.search(r"週次[：:]\s*(\d+)", system)
        if week_match:
            week = int(week_match.group(1))

        # Extract topic
        topic = ""
        topic_match = re.search(r"主題[：:]\s*(.+)", system)
        if topic_match:
            topic = topic_match.group(1).strip()

        # Determine if homework mode
        is_homework = "作業模式" in system or "作業" in system[:100]

        # Build conversation history
        history = [{"role": m.role, "content": m.content} for m in messages]

        # Run the FULL 42-layer NLP pipeline
        ctx = NLPContext(
            user_message=user_msg,
            conversation_history=history,
            week=week,
            topic=topic,
            is_homework=is_homework,
        )
        ctx = run_pipeline(ctx, FULL_PIPELINE)

        logger.info(
            "NLP pipeline: %d layers in %.0fms | intent=%s emotion=%s level=%s",
            len(ctx.layers_executed), ctx.total_processing_ms,
            ctx.intent, ctx.emotion, ctx.student_level,
        )

        return ctx.response

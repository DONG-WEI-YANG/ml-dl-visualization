"""Local NLP provider — no external API needed.

Uses a 7-layer NLP pipeline for question analysis and response generation.
Works offline, zero cost, always available.
"""

import re
from .base import LLMProvider, LLMMessage, LLMResponse
from app.nlp.pipeline import NLPContext
from app.nlp.intent import detect_intent
from app.nlp.emotion import detect_emotion
from app.nlp.difficulty import assess_difficulty
from app.nlp.topic import extract_topics
from app.nlp.context_tracker import track_conversation
from app.nlp.reranker import retrieve_and_rerank
from app.nlp.response import assemble_response


def run_pipeline(user_message: str, history: list[dict], week: int, topic: str, is_homework: bool) -> NLPContext:
    """Run all 7 NLP layers in sequence."""
    ctx = NLPContext(
        user_message=user_message,
        conversation_history=history,
        week=week,
        topic=topic,
        is_homework=is_homework,
    )

    # Layer 1-7 in sequence, each enriching the shared context
    ctx = detect_intent(ctx)        # L1: What type of question?
    ctx = detect_emotion(ctx)       # L2: How is the student feeling?
    ctx = assess_difficulty(ctx)    # L3: What's their level?
    ctx = extract_topics(ctx)       # L4: What concepts are involved?
    ctx = track_conversation(ctx)   # L5: Multi-turn context
    ctx = retrieve_and_rerank(ctx)  # L6: Find relevant curriculum
    ctx = assemble_response(ctx)    # L7: Build adaptive response

    return ctx


class LocalProvider(LLMProvider):
    """Local NLP provider using 7-layer pipeline. No API needed."""

    def __init__(self):
        self.model_name = "local-nlp-v2"

    async def chat(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        response_text = self._generate(messages, system)
        return LLMResponse(content=response_text, model=self.model_name)

    async def stream(self, messages: list[LLMMessage], system: str = ""):
        """Simulate streaming by yielding chunks of the full response."""
        response_text = self._generate(messages, system)
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

        # Run the full NLP pipeline
        ctx = run_pipeline(user_msg, history, week, topic, is_homework)

        return ctx.response

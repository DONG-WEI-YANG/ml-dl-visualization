import logging
import asyncio
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from pydantic import BaseModel
from app.llm.factory import create_llm_provider
from app.llm.tutor import AITutor
from app.llm.base import LLMMessage
from app.config import settings
from app.db import get_db, get_setting
from app.auth.utils import decode_token
from app.auth.dependencies import get_current_user
from app.llm.quick_answer import build_quick_answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM"])


def _get_provider():
    """Create LLM provider from admin-configured settings.
    Falls back to local NLP if the configured provider lacks API keys."""
    provider = get_setting("llm_provider", "local")
    model = get_setting("llm_model", "")
    try:
        p = create_llm_provider(provider=provider, model=model or None)
        # Verify API-based providers have keys
        if provider == "anthropic" and not settings.anthropic_api_key:
            logger.warning("No Anthropic API key, falling back to local NLP")
            return create_llm_provider(provider="local")
        if provider == "openai" and not settings.openai_api_key:
            logger.warning("No OpenAI API key, falling back to local NLP")
            return create_llm_provider(provider="local")
        return p
    except Exception:
        return create_llm_provider(provider="local")


def _rag_enabled() -> bool:
    return get_setting("rag_enabled", "true").lower() == "true"


def _make_tutor() -> AITutor:
    return AITutor(_get_provider(), use_rag=_rag_enabled())


async def _stream_chat_events(messages, week: int, topic: str, mode: str):
    """Orchestrate a bounded draft followed by the fully verified answer."""
    started = time.monotonic()
    draft_ms = None
    yield {"type": "status", "stage": "analyzing"}
    question = next((m.content for m in reversed(messages) if m.role == "user"), "")
    try:
        draft = await asyncio.wait_for(
            asyncio.to_thread(build_quick_answer, question, week, topic),
            timeout=0.8,
        )
        draft_ms = round((time.monotonic() - started) * 1000)
        if draft:
            yield {"type": "draft", "content": _safe_text(draft), "elapsed_ms": draft_ms}
    except Exception as exc:
        logger.warning("AI draft skipped week=%d reason=%s", week, type(exc).__name__)

    yield {"type": "status", "stage": "verifying"}
    tutor = _make_tutor()
    try:
        async for chunk in tutor.ask_stream(messages, week=week, topic=topic, mode=mode):
            yield {"type": "refinement", "content": _safe_text(chunk)}
        total_ms = round((time.monotonic() - started) * 1000)
        logger.info("AI stream week=%d draft_ms=%s total_ms=%d", week, draft_ms, total_ms)
        yield {"type": "done", "elapsed_ms": total_ms, "draft_ms": draft_ms}
    except Exception as exc:
        logger.error("AI refinement failed week=%d error=%s", week, type(exc).__name__)
        yield {"type": "error", "stage": "refinement", "content": "完整回答暫時無法完成，請稍後重試。"}


class ChatRequest(BaseModel):
    messages: list[LLMMessage]
    week: int = 1
    topic: str = ""
    mode: str = "tutor"  # tutor | homework


@router.get("/model-info")
async def get_model_info():
    """Return the currently configured LLM model (for display only)."""
    return {
        "provider": get_setting("llm_provider", "anthropic"),
        "model": get_setting("llm_model", ""),
    }


def _safe_text(text: str) -> str:
    """Remove Unicode surrogates that crash JSON serialization."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


@router.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    try:
        provider = _get_provider()
        tutor = AITutor(provider, use_rag=_rag_enabled())
        response = await tutor.ask(req.messages, week=req.week, topic=req.topic, mode=req.mode)
        return {"response": _safe_text(response.content), "model": response.model}
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        return {"response": f"抱歉，處理你的問題時發生錯誤。請稍後再試。\n\n錯誤資訊：{type(e).__name__}", "model": "error"}


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket, token: str = Query(default="")):
    payload = decode_token(token) if token else None
    if not payload:
        await websocket.close(code=4401, reason="需要登入")
        return
    conn = get_db()
    user = conn.execute(
        "SELECT id FROM users WHERE id = ? AND is_active = 1 AND deleted_at IS NULL",
        (payload["sub"],),
    ).fetchone()
    conn.close()
    if not user:
        await websocket.close(code=4401, reason="需要登入")
        return
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            messages = [LLMMessage(**m) for m in data.get("messages", [])]
            week = data.get("week", 1)
            topic = data.get("topic", "")
            mode = data.get("mode", "tutor")
            async for event in _stream_chat_events(messages, week, topic, mode):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass

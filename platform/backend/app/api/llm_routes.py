import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
from app.llm.factory import create_llm_provider
from app.llm.tutor import AITutor
from app.llm.base import LLMMessage
from app.config import settings
from app.db import get_setting
from app.auth.utils import decode_token

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


@router.post("/chat")
async def chat(req: ChatRequest):
    provider = _get_provider()
    tutor = AITutor(provider, use_rag=_rag_enabled())
    response = await tutor.ask(req.messages, week=req.week, topic=req.topic, mode=req.mode)
    return {"response": response.content, "model": response.model}


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket, token: str = Query(default="")):
    # Validate token before accepting connection
    if token:
        payload = decode_token(token)
        if not payload:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            messages = [LLMMessage(**m) for m in data.get("messages", [])]
            week = data.get("week", 1)
            topic = data.get("topic", "")
            mode = data.get("mode", "tutor")
            provider = _get_provider()
            tutor = AITutor(provider, use_rag=_rag_enabled())
            async for chunk in tutor.ask_stream(messages, week=week, topic=topic, mode=mode):
                await websocket.send_json({"type": "chunk", "content": chunk})
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass

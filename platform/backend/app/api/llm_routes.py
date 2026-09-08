import logging
import asyncio
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from pydantic import BaseModel
from app.llm.factory import ProviderResolution, resolve_llm_provider
from app.llm.tutor import AITutor
from app.llm.base import LLMMessage
from app.config import settings
from app.db import get_db, get_setting
from app.auth.utils import decode_token
from app.auth.dependencies import get_current_user, require_admin
from app.llm.quick_answer import build_quick_answer
from app.analytics.tracker import record_event
from app.analytics.models import LearningEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM"])


def _get_provider_resolution() -> ProviderResolution:
    """Resolve the configured provider and any safe local fallback."""
    provider = get_setting("llm_provider", "local")
    model = get_setting("llm_model", "")
    resolution = resolve_llm_provider(provider=provider, model=model or None)
    if resolution.reason:
        logger.warning(
            "LLM provider fallback configured=%s effective=%s reason=%s",
            resolution.configured_provider,
            resolution.effective_provider,
            resolution.reason,
        )
    return resolution


def _get_provider():
    return _get_provider_resolution().provider


def _rag_enabled() -> bool:
    return get_setting("rag_enabled", "true").lower() == "true"


def _make_tutor() -> AITutor:
    return AITutor(_get_provider(), use_rag=_rag_enabled())


async def _stream_chat_events(
    messages,
    week: int,
    topic: str,
    mode: str,
    student_id: str | None = None,
):
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
        async for chunk in tutor.ask_stream(
            messages,
            week=week,
            topic=topic,
            mode=mode,
            student_id=student_id,
        ):
            yield {"type": "refinement", "content": _safe_text(chunk)}
        total_ms = round((time.monotonic() - started) * 1000)
        if student_id:
            record_event(LearningEvent(student_id=student_id, week=week,
                                       event_type='llm_chat', topic=topic))
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
    """Return safe configured and effective model metadata."""
    resolution = _get_provider_resolution()
    return _resolution_payload(resolution)


def _resolution_payload(resolution: ProviderResolution) -> dict:
    return {
        "provider": resolution.configured_provider,
        "model": resolution.configured_model,
        "configured_provider": resolution.configured_provider,
        "configured_model": resolution.configured_model,
        "effective_provider": resolution.effective_provider,
        "effective_model": resolution.effective_model,
        "status": resolution.status,
        "reason": resolution.reason,
        "runtime_warnings": list(resolution.runtime_warnings),
    }


@router.get("/diagnostics")
async def llm_diagnostics(
    probe: bool = Query(False),
    _admin: dict = Depends(require_admin),
):
    """Report truthful provider resolution and optionally execute a minimal probe."""
    resolution = _get_provider_resolution()
    payload = _resolution_payload(resolution)
    payload["probe"] = {"attempted": False}
    if not probe:
        return payload

    started = time.monotonic()
    timeout_seconds = 60 if resolution.effective_provider == "local" else 20
    try:
        response = await asyncio.wait_for(
            resolution.provider.chat(
                [LLMMessage(role="user", content="請只回答：2")],
                system="這是連線健康檢查。請計算 1+1。",
            ),
            timeout=timeout_seconds,
        )
        if not response.content.strip():
            raise ValueError("empty_response")
        # A successful response proves availability, but does not erase known
        # model/runtime degradation reported by the resolver.
        payload["status"] = resolution.status if resolution.status == "degraded" else "ready"
        payload["probe"] = {
            "attempted": True,
            "ok": True,
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
    except Exception as exc:
        logger.warning("LLM diagnostic probe failed: %s", type(exc).__name__)
        payload["status"] = "error"
        payload["probe"] = {
            "attempted": True,
            "ok": False,
            "reason": "provider_unreachable",
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
    return payload


def _safe_text(text: str) -> str:
    """Remove Unicode surrogates that crash JSON serialization."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


@router.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    try:
        provider = _get_provider()
        tutor = AITutor(provider, use_rag=_rag_enabled())
        response = await tutor.ask(
            req.messages,
            week=req.week,
            topic=req.topic,
            mode=req.mode,
            student_id=str(user["id"]),
        )
        record_event(LearningEvent(student_id=str(user['id']), week=req.week,
                                   event_type='llm_chat', topic=req.topic))
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
            async for event in _stream_chat_events(
                messages,
                week,
                topic,
                mode,
                student_id=str(user["id"]),
            ):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass

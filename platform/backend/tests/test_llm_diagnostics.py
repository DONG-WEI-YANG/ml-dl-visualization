import threading

import pytest
from fastapi.testclient import TestClient

from app.db import get_setting, set_setting
from app.llm.factory import resolve_llm_provider
from app.llm.local_provider import LocalProvider
from app.llm.base import LLMMessage
from app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_llm_settings(monkeypatch):
    """Keep diagnostics deterministic regardless of settings changed by other tests."""
    original_provider = get_setting("llm_provider", "local")
    original_model = get_setting("llm_model", "local-nlp")
    set_setting("llm_provider", "local")
    set_setting("llm_model", "local-nlp")
    monkeypatch.setattr("app.llm.factory.get_local_runtime_warnings", lambda: ())
    try:
        yield
    finally:
        set_setting("llm_provider", original_provider)
        set_setting("llm_model", original_model)


def _admin_headers() -> dict[str, str]:
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_local_provider_resolution_reports_same_effective_provider():
    resolution = resolve_llm_provider("local", "local-nlp")

    assert resolution.configured_provider == "local"
    assert resolution.effective_provider == "local"
    assert resolution.effective_model == "local-nlp-v3"
    assert resolution.status == "ready"
    assert resolution.reason is None


def test_missing_openai_key_resolves_to_explicit_local_fallback(monkeypatch):
    monkeypatch.setattr("app.llm.factory.settings.openai_api_key", "")

    resolution = resolve_llm_provider("openai", "gpt-test")

    assert resolution.configured_provider == "openai"
    assert resolution.configured_model == "gpt-test"
    assert resolution.effective_provider == "local"
    assert resolution.status == "degraded"
    assert resolution.reason == "missing_api_key"


def test_local_runtime_limitations_are_reported_as_degraded(monkeypatch):
    monkeypatch.setattr(
        "app.llm.factory.get_local_runtime_warnings",
        lambda: ("sklearn_model_version_mismatch", "semantic_model_uncached"),
    )

    resolution = resolve_llm_provider("local", "local-nlp")

    assert resolution.status == "degraded"
    assert resolution.reason == "runtime_degraded"
    assert resolution.runtime_warnings == (
        "sklearn_model_version_mismatch",
        "semantic_model_uncached",
    )


def test_unknown_provider_resolves_to_explicit_local_fallback():
    resolution = resolve_llm_provider("not-a-provider", "mystery")

    assert resolution.configured_provider == "not-a-provider"
    assert resolution.effective_provider == "local"
    assert resolution.status == "degraded"
    assert resolution.reason == "unknown_provider"


def test_diagnostics_requires_admin_and_never_returns_secrets():
    assert client.get("/api/llm/diagnostics").status_code == 401

    response = client.get("/api/llm/diagnostics", headers=_admin_headers())

    assert response.status_code == 200
    data = response.json()
    assert data["configured_provider"] == "local"
    assert data["effective_provider"] == "local"
    serialized = response.text.lower()
    assert "api_key" not in serialized
    assert "admin123" not in serialized


def test_model_info_uses_effective_resolution_metadata():
    response = client.get("/api/llm/model-info")

    assert response.status_code == 200
    assert response.json()["effective_provider"] == "local"
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_local_provider_generation_does_not_block_event_loop(monkeypatch):
    provider = LocalProvider()
    generation_threads: list[int] = []

    def slow_generate(_messages, _system):
        generation_threads.append(threading.get_ident())
        return "ok"

    monkeypatch.setattr(provider, "_generate", slow_generate)
    event_loop_thread = threading.get_ident()

    response = await provider.chat([LLMMessage(role="user", content="hello")])

    assert response.content == "ok"
    assert generation_threads[0] != event_loop_thread

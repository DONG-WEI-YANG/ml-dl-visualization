from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"starting", "ready", "degraded"}
    assert data["database"] == "connected"
    assert data["rag"] in {"pending", "indexing", "ready", "error"}
    assert isinstance(data["uptime_seconds"], (int, float))


def test_health_does_not_run_rag_work(monkeypatch):
    def fail_if_called():
        raise AssertionError("health must not inspect or initialize RAG")

    monkeypatch.setattr("app.main._auto_ingest_curriculum", fail_if_called)
    response = client.get("/health")
    assert response.status_code == 200

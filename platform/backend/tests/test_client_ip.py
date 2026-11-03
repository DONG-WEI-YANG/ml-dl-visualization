"""Behind the HF proxy every request arrives from the proxy's address, so the
rate limiter and audit log must use the forwarded client address."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main
from app.net import client_ip

client = TestClient(main.app)


def _req(headers=None, host="10.0.0.1"):
    return SimpleNamespace(headers=headers or {}, client=SimpleNamespace(host=host))


def test_uses_rightmost_forwarded_address():
    # The proxy appends the address it saw; anything to the left is client-supplied.
    assert client_ip(_req({"x-forwarded-for": "6.6.6.6, 203.0.113.7"})) == "203.0.113.7"
    assert client_ip(_req({"x-forwarded-for": "203.0.113.7"})) == "203.0.113.7"


def test_falls_back_to_socket_address():
    assert client_ip(_req()) == "10.0.0.1"
    assert client_ip(_req({"x-forwarded-for": "  "})) == "10.0.0.1"
    assert client_ip(SimpleNamespace(headers={}, client=None)) == ""


def test_rate_limit_is_per_forwarded_client(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 3)
    a = {"X-Forwarded-For": "198.51.100.1"}
    b = {"X-Forwarded-For": "198.51.100.2"}
    for _ in range(3):
        assert client.get("/health", headers=a).status_code == 200
    assert client.get("/health", headers=a).status_code == 429
    # A different student behind the same proxy is not blocked.
    assert client.get("/health", headers=b).status_code == 200


def test_spoofed_left_entries_do_not_evade_limit(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    for fake in ("1.1.1.1", "2.2.2.2"):
        assert client.get("/health", headers={"X-Forwarded-For": f"{fake}, 198.51.100.9"}).status_code == 200
    assert client.get("/health", headers={"X-Forwarded-For": "3.3.3.3, 198.51.100.9"}).status_code == 429

"""CSV batch user import."""
from fastapi.testclient import TestClient
from app.main import app
from app.db import get_db

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_import_creates_students_with_initial_passwords():
    resp = client.post(
        "/api/admin/users/import",
        json={"semester": "115-1", "rows": [
            {"username": "s115001", "display_name": "王小明"},
            {"username": "s115002", "display_name": "李小華", "email": "b@x.tw"},
        ]},
        headers=_admin(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["created"]) == 2
    assert data["skipped"] == []
    pw = data["created"][0]["initial_password"]
    assert len(pw) >= 12
    # new account can login with the returned initial password and is flagged
    login = client.post("/api/auth/login", json={"username": "s115001", "password": pw})
    assert login.status_code == 200
    assert login.json()["user"]["must_change_password"] is True
    assert login.json()["user"]["role"] == "student"
    assert login.json()["user"]["semester"] == "115-1"


def test_import_skips_duplicates_and_bad_rows():
    client.post(
        "/api/admin/users/import",
        json={"rows": [{"username": "dupuser"}]},
        headers=_admin(),
    )
    resp = client.post(
        "/api/admin/users/import",
        json={"rows": [{"username": "dupuser"}, {"username": ""}, {"username": "okuser"}]},
        headers=_admin(),
    )
    data = resp.json()
    assert [c["username"] for c in data["created"]] == ["okuser"]
    reasons = {s["username"]: s["reason"] for s in data["skipped"]}
    assert "dupuser" in reasons and "" in reasons


def test_import_audited_with_count():
    client.post(
        "/api/admin/users/import",
        json={"rows": [{"username": "audituser1"}]},
        headers=_admin(),
    )
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_logs WHERE action='user.import' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    assert '"created": 1' in row["detail"]

"""Regression tests for backend review findings: soft-deleted users must be
excluded from update/roster/assign paths, and must_change_password must be
exposed in admin user responses."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _admin():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_user(username, role="student"):
    headers = _admin()
    resp = client.post(
        "/api/auth/register",
        json={"username": username, "password": "somepass1", "role": role},
        headers=headers,
    )
    return resp.json()["id"]


def test_update_soft_deleted_user_returns_404():
    uid = _create_user("rf_upd_user1")
    headers = _admin()
    del_resp = client.delete(f"/api/admin/users/{uid}", headers=headers)
    assert del_resp.status_code == 200
    resp = client.put(f"/api/admin/users/{uid}", json={"email": "x@y.tw"}, headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "使用者不存在"


def test_teacher_roster_excludes_soft_deleted_student():
    headers = _admin()
    teacher_id = _create_user("rf_teacher1", role="teacher")
    student_id = _create_user("rf_student1", role="student")
    assign_resp = client.post(f"/api/admin/teachers/{teacher_id}/students/{student_id}", headers=headers)
    assert assign_resp.status_code == 200
    del_resp = client.delete(f"/api/admin/users/{student_id}", headers=headers)
    assert del_resp.status_code == 200
    roster = client.get(f"/api/admin/teachers/{teacher_id}/students", headers=headers)
    assert roster.status_code == 200
    assert all(u["id"] != student_id for u in roster.json())


def test_import_sets_must_change_password_flag_in_admin_listing():
    headers = _admin()
    resp = client.post(
        "/api/admin/users/import",
        json={"rows": [{"username": "rf_import_user1"}]},
        headers=headers,
    )
    assert resp.status_code == 200
    listing = client.get("/api/admin/users", headers=headers)
    assert listing.status_code == 200
    entry = next(u for u in listing.json() if u["username"] == "rf_import_user1")
    assert entry["must_change_password"] is True


def test_assign_soft_deleted_student_returns_404():
    headers = _admin()
    teacher_id = _create_user("rf_teacher2", role="teacher")
    student_id = _create_user("rf_student2", role="student")
    del_resp = client.delete(f"/api/admin/users/{student_id}", headers=headers)
    assert del_resp.status_code == 200
    resp = client.post(f"/api/admin/teachers/{teacher_id}/students/{student_id}", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "學生不存在"


def test_assign_soft_deleted_teacher_returns_404():
    headers = _admin()
    teacher_id = _create_user("rf_teacher3", role="teacher")
    student_id = _create_user("rf_student3", role="student")
    del_resp = client.delete(f"/api/admin/users/{teacher_id}", headers=headers)
    assert del_resp.status_code == 200
    resp = client.post(f"/api/admin/teachers/{teacher_id}/students/{student_id}", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "教師不存在"

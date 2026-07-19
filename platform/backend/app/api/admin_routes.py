import secrets
import sqlite3

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from app.auth.models import UserOut, UserUpdate, UserCreate, ImportRow, ImportRequest
from app.auth.utils import hash_password
from app.auth.dependencies import require_admin, require_teacher_or_admin
from app.db import get_db, get_all_settings, set_setting
from app.llm.factory import list_available_providers
from app.nlp.trainer import train_models
from app.audit import log_audit

router = APIRouter(prefix="/api/admin", tags=["Admin"])


def _user_out(row) -> UserOut:
    return UserOut(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        email=row["email"],
        semester=row["semester"],
        role=row["role"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        must_change_password=bool(row["must_change_password"]),
    )


# ── User CRUD (admin only) ──


@router.get("/users", response_model=list[UserOut])
async def list_users(
    role: str | None = Query(None),
    semester: str | None = Query(None),
    admin: dict = Depends(require_admin),
):
    conn = get_db()
    conditions = ["deleted_at IS NULL"]
    params = []
    if role:
        conditions.append("role = ?")
        params.append(role)
    if semester:
        conditions.append("semester = ?")
        params.append(semester)
    where = f" WHERE {' AND '.join(conditions)}"
    rows = conn.execute(f"SELECT * FROM users{where} ORDER BY id", params).fetchall()
    conn.close()
    return [_user_out(r) for r in rows]


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int, admin: dict = Depends(require_admin)):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ? AND deleted_at IS NULL", (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="使用者不存在")
    return _user_out(row)


@router.put("/users/{user_id}", response_model=UserOut)
async def update_user(user_id: int, data: UserUpdate, request: Request, admin: dict = Depends(require_admin)):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ? AND deleted_at IS NULL", (user_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="使用者不存在")

    updates = []
    params = []
    if data.display_name is not None:
        updates.append("display_name = ?")
        params.append(data.display_name)
    if data.email is not None:
        updates.append("email = ?")
        params.append(data.email)
    if data.role is not None:
        if data.role not in ("admin", "teacher", "student"):
            conn.close()
            raise HTTPException(status_code=400, detail="無效角色")
        updates.append("role = ?")
        params.append(data.role)
    if data.is_active is not None:
        updates.append("is_active = ?")
        params.append(int(data.is_active))
    if data.semester is not None:
        updates.append("semester = ?")
        params.append(data.semester)
    if data.password is not None:
        updates.append("password_hash = ?")
        params.append(hash_password(data.password))
        updates.append("must_change_password = 1")

    if updates:
        updates.append("updated_at = datetime('now')")
        params.append(user_id)
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

    updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()

    changed = [f for f in ("display_name", "email", "role", "is_active", "semester")
               if getattr(data, f) is not None]
    ip = request.client.host if request.client else ""
    if changed:
        log_audit("user.update", actor=admin, target_type="user", target_id=user_id,
                  detail={"fields": changed}, ip=ip)
    if data.password is not None:
        log_audit("user.password_reset", actor=admin, target_type="user", target_id=user_id, ip=ip)

    return _user_out(updated)


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, request: Request, user=Depends(require_admin)):
    if user["id"] == user_id:
        raise HTTPException(status_code=400, detail="不能刪除自己的帳號")
    db = get_db()
    cursor = db.execute(
        "UPDATE users SET is_active = 0, deleted_at = datetime('now'), "
        "updated_at = datetime('now') WHERE id = ? AND deleted_at IS NULL",
        (user_id,),
    )
    db.commit()
    db.close()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="使用者不存在")
    ip = request.client.host if request.client else ""
    log_audit("user.delete", actor=user, target_type="user", target_id=user_id, ip=ip)
    return {"deleted": True}


@router.post("/users/import")
async def import_users(req: ImportRequest, request: Request, admin: dict = Depends(require_admin)):
    from app.db import get_setting
    semester = req.semester or get_setting("current_semester", "")
    conn = get_db()
    ip = request.client.host if request.client else ""
    created, skipped, restored = [], [], []
    restored_ids = []  # audited after commit — log_audit opens its own connection
    for row in req.rows:
        username = row.username.strip()
        if not username:
            skipped.append({"username": row.username, "reason": "帳號為空"})
            continue
        existing = conn.execute(
            "SELECT id, deleted_at FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            if existing["deleted_at"] is not None:
                # Re-enrolling student: restore the archived account so their
                # learning history (keyed by user id) stays linked.
                initial_password = secrets.token_urlsafe(9)  # 12 chars
                conn.execute(
                    "UPDATE users SET deleted_at = NULL, is_active = 1, password_hash = ?, "
                    "must_change_password = 1, semester = ?, updated_at = datetime('now') "
                    "WHERE id = ?",
                    (hash_password(initial_password), semester, existing["id"]),
                )
                restored.append({"username": username, "initial_password": initial_password})
                restored_ids.append(existing["id"])
            else:
                skipped.append({"username": username, "reason": "帳號已存在"})
            continue
        initial_password = secrets.token_urlsafe(9)  # 12 chars
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, email, role, "
            "semester, must_change_password) VALUES (?, ?, ?, ?, 'student', ?, 1)",
            (username, hash_password(initial_password),
             row.display_name.strip() or username, row.email.strip(), semester),
        )
        created.append({"username": username, "initial_password": initial_password})
    conn.commit()
    conn.close()
    for user_id in restored_ids:
        log_audit("user.restore", actor=admin, target_type="user",
                  target_id=user_id, detail={"via": "import"}, ip=ip)
    log_audit("user.import", actor=admin, target_type="user",
              detail={"created": len(created), "skipped": len(skipped),
                       "restored": len(restored), "semester": semester}, ip=ip)
    return {"created": created, "skipped": skipped, "restored": restored}


@router.post("/semesters/{semester}/archive")
async def archive_semester(semester: str, request: Request, admin: dict = Depends(require_admin)):
    conn = get_db()
    cursor = conn.execute(
        "UPDATE users SET is_active = 0, updated_at = datetime('now') "
        "WHERE semester = ? AND role = 'student' AND is_active = 1 AND deleted_at IS NULL",
        (semester,),
    )
    conn.commit()
    conn.close()
    ip = request.client.host if request.client else ""
    log_audit("semester.archive", actor=admin, target_type="semester", target_id=semester,
              detail={"archived": cursor.rowcount}, ip=ip)
    return {"archived": cursor.rowcount}


# ── Teacher-Student assignment (admin only) ──


@router.post("/teachers/{teacher_id}/students/{student_id}")
async def assign_student_to_teacher(teacher_id: int, student_id: int, request: Request, admin: dict = Depends(require_admin)):
    conn = get_db()
    teacher = conn.execute(
        "SELECT * FROM users WHERE id = ? AND role = 'teacher' AND deleted_at IS NULL", (teacher_id,)
    ).fetchone()
    student = conn.execute(
        "SELECT * FROM users WHERE id = ? AND role = 'student' AND deleted_at IS NULL", (student_id,)
    ).fetchone()
    if not teacher:
        conn.close()
        raise HTTPException(status_code=404, detail="教師不存在")
    if not student:
        conn.close()
        raise HTTPException(status_code=404, detail="學生不存在")
    conn.execute(
        "INSERT OR IGNORE INTO teacher_students (teacher_id, student_id) VALUES (?, ?)",
        (teacher_id, student_id),
    )
    conn.commit()
    conn.close()
    ip = request.client.host if request.client else ""
    log_audit("teacher_student.assign", actor=admin, target_type="teacher_student",
              target_id=f"{teacher_id}:{student_id}", ip=ip)
    return {"status": "assigned"}


@router.delete("/teachers/{teacher_id}/students/{student_id}")
async def remove_student_from_teacher(teacher_id: int, student_id: int, request: Request, admin: dict = Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM teacher_students WHERE teacher_id = ? AND student_id = ?", (teacher_id, student_id))
    conn.commit()
    conn.close()
    ip = request.client.host if request.client else ""
    log_audit("teacher_student.remove", actor=admin, target_type="teacher_student",
              target_id=f"{teacher_id}:{student_id}", ip=ip)
    return {"status": "removed"}


# ── Teacher views (teacher or admin) ──


@router.get("/teachers/{teacher_id}/students", response_model=list[UserOut])
async def get_teacher_students(teacher_id: int, user: dict = Depends(require_teacher_or_admin)):
    if user["role"] == "teacher" and user["id"] != teacher_id:
        raise HTTPException(status_code=403, detail="只能查看自己的學生")
    conn = get_db()
    rows = conn.execute(
        """SELECT u.* FROM users u
           JOIN teacher_students ts ON u.id = ts.student_id
           WHERE ts.teacher_id = ? AND u.deleted_at IS NULL ORDER BY u.id""",
        (teacher_id,),
    ).fetchall()
    conn.close()
    return [_user_out(r) for r in rows]


@router.get("/teachers", response_model=list[UserOut])
async def list_teachers(user: dict = Depends(require_teacher_or_admin)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM users WHERE role = 'teacher' AND is_active = 1 ORDER BY id").fetchall()
    conn.close()
    return [_user_out(r) for r in rows]


# ── System settings (admin only) ──


@router.get("/settings")
async def get_settings(admin: dict = Depends(require_admin)):
    """Get all system settings including LLM config."""
    settings = get_all_settings()
    providers = list_available_providers()
    return {"settings": settings, "available_providers": providers}


@router.put("/settings")
async def update_settings(data: dict, request: Request, admin: dict = Depends(require_admin)):
    """Update system settings. Accepts key-value pairs."""
    allowed_keys = {"llm_provider", "llm_model", "rag_enabled", "rag_top_k", "current_semester"}
    for key, value in data.items():
        if key not in allowed_keys:
            raise HTTPException(status_code=400, detail=f"不允許的設定項: {key}")
        set_setting(key, str(value))
    ip = request.client.host if request.client else ""
    log_audit("settings.update", actor=admin, target_type="setting",
              detail={"keys": sorted(data.keys())}, ip=ip)
    return {"status": "updated", "settings": get_all_settings()}


# ── NLP Model Training (admin only) ──


@router.post("/train-nlp")
async def train_nlp_models(request: Request, admin: dict = Depends(require_admin)):
    """Train all NLP models (intent, emotion, corpus TF-IDF)."""
    try:
        results = train_models()

        # Reload models in all NLP layers
        from app.nlp.intent import reload_model as reload_intent
        from app.nlp.emotion import reload_model as reload_emotion
        from app.nlp.topic import reload_model as reload_topic
        from app.nlp.reranker import reload_model as reload_reranker

        reload_intent()
        reload_emotion()
        reload_topic()
        reload_reranker()

        ip = request.client.host if request.client else ""
        log_audit("nlp.train", actor=admin, target_type="nlp", ip=ip)
        return {"status": "trained", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"訓練失敗: {str(e)}")


# --- Web Enrichment ---


@router.post("/enrichment/trigger")
async def trigger_enrichment(request: Request, admin: dict = Depends(require_admin)):
    """Manually trigger web enrichment. Admin only."""
    from app.rag.web_enricher import enrich_from_web
    result = await enrich_from_web()
    ip = request.client.host if request.client else ""
    log_audit("enrichment.trigger", actor=admin, target_type="nlp", ip=ip)
    return {"status": "ok", **result}


@router.get("/enrichment/history")
async def enrichment_history(admin: dict = Depends(require_admin)):
    """Get enrichment run history. Admin only."""
    from app.rag.store import get_enrichment_history
    return {"history": get_enrichment_history()}


# --- Quiz Management ---


@router.get("/quiz/questions")
async def admin_list_questions(week: int | None = None, _user=Depends(require_admin)):
    from app.quiz.questions import list_all_questions
    return {"questions": list_all_questions(week)}


@router.post("/quiz/questions")
async def admin_create_question(body: dict, request: Request, _user=Depends(require_admin)):
    from app.quiz.questions import create_question
    required = {"id", "week", "question", "options", "answer"}
    if not required.issubset(body.keys()):
        raise HTTPException(400, f"缺少必要欄位：{required - body.keys()}")
    if not isinstance(body["options"], list) or len(body["options"]) < 2:
        raise HTTPException(400, "選項至少需要 2 個")
    if not isinstance(body["answer"], int) or body["answer"] >= len(body["options"]):
        raise HTTPException(400, "答案索引超出選項範圍")
    try:
        q = create_question(body)
    except sqlite3.IntegrityError:
        raise HTTPException(400, "題目 ID 已存在")
    except Exception as e:
        raise HTTPException(400, f"建立題目失敗：{e}")
    ip = request.client.host if request.client else ""
    log_audit("quiz.create", actor=_user, target_type="quiz_question", target_id=body["id"], ip=ip)
    return {"question": q}


@router.put("/quiz/questions/{question_id}")
async def admin_update_question(question_id: str, body: dict, request: Request, _user=Depends(require_admin)):
    from app.quiz.questions import update_question
    q = update_question(question_id, body)
    if not q:
        raise HTTPException(404, "題目不存在")
    ip = request.client.host if request.client else ""
    log_audit("quiz.update", actor=_user, target_type="quiz_question", target_id=question_id, ip=ip)
    return {"question": q}


@router.delete("/quiz/questions/{question_id}")
async def admin_delete_question(question_id: str, request: Request, _user=Depends(require_admin)):
    from app.quiz.questions import delete_question
    if not delete_question(question_id):
        raise HTTPException(404, "題目不存在")
    ip = request.client.host if request.client else ""
    log_audit("quiz.delete", actor=_user, target_type="quiz_question", target_id=question_id, ip=ip)
    return {"deleted": True}

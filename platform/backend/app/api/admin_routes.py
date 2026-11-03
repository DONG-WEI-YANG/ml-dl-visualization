import secrets
import sqlite3

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from app.auth.models import UserOut, UserUpdate, UserCreate, ImportRow, ImportRequest, BulkUserAction
from app.auth.utils import hash_password
from app.auth.dependencies import require_admin, require_teacher_or_admin
from app.db import get_db, get_all_settings, set_setting, sync_enrollment
from app.llm.factory import list_available_providers
from app.nlp.trainer import train_models
from app.audit import log_audit
from app.net import client_ip

router = APIRouter(prefix="/api/admin", tags=["Admin"])


def _user_out(row) -> UserOut:
    return UserOut(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        email=row["email"],
        semester=row["semester"],
        class_name=row["class_name"],
        role=row["role"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        must_change_password=bool(row["must_change_password"]),
        approval_status=row["approval_status"],
    )


# ── User CRUD (admin only) ──


@router.get("/users", response_model=list[UserOut])
async def list_users(
    role: str | None = Query(None),
    semester: str | None = Query(None),
    class_name: str | None = Query(None),
    academic_year: str | None = Query(None),
    q: str | None = Query(None, max_length=100),
    status: str | None = Query(None, pattern="^(active|inactive|pending|rejected)$"),
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
    if class_name:
        conditions.append('class_name = ?')
        params.append(class_name)
    if academic_year:
        conditions.append("substr(semester, 1, instr(semester, '-') - 1) = ?")
        params.append(academic_year)
    if q and q.strip():
        # Escape LIKE wildcards so a search for "s_01" matches literally.
        term = q.strip().replace("!", "!!").replace("%", "!%").replace("_", "!_")
        like = f"%{term}%"
        conditions.append(
            "(username LIKE ? ESCAPE '!' OR display_name LIKE ? ESCAPE '!' "
            "OR email LIKE ? ESCAPE '!' OR class_name LIKE ? ESCAPE '!')"
        )
        params.extend([like] * 4)
    if status in ("pending", "rejected"):
        conditions.append("approval_status = ?")
        params.append(status)
    elif status == "active":
        conditions.append("is_active = 1 AND approval_status = 'approved'")
    elif status == "inactive":
        conditions.append("is_active = 0 AND approval_status = 'approved'")
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
    sync_enrollment(conn, user_id)
    if data.class_name is not None:
        updates.append('class_name = ?')
        params.append(data.class_name.strip())
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
        if (data.password is not None
            or (data.role is not None and data.role != row['role'])
            or (data.is_active is not None and data.is_active != bool(row['is_active']))):
            updates.append('session_version = session_version + 1')
        updates.append("updated_at = datetime('now')")
        params.append(user_id)
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
    sync_enrollment(conn, user_id)
    conn.commit()

    updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()

    changed = [f for f in ("display_name", "email", "role", "is_active", "semester", "class_name")
               if getattr(data, f) is not None]
    ip = client_ip(request)
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
        "UPDATE users SET is_active = 0, deleted_at = datetime('now'), session_version = session_version + 1, "
        "updated_at = datetime('now') WHERE id = ? AND deleted_at IS NULL",
        (user_id,),
    )
    db.commit()
    db.close()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="使用者不存在")
    ip = client_ip(request)
    log_audit("user.delete", actor=user, target_type="user", target_id=user_id, ip=ip)
    return {"deleted": True}


@router.post("/users/import")
async def import_users(req: ImportRequest, request: Request, admin: dict = Depends(require_admin)):
    from app.db import get_setting
    semester = req.semester or get_setting("current_semester", "")
    conn = get_db()
    ip = client_ip(request)
    created, skipped, restored = [], [], []
    restored_ids = []  # audited after commit — log_audit opens its own connection
    for row in req.rows:
        username = row.username.strip()
        if not username:
            skipped.append({"username": row.username, "reason": "帳號為空"})
            continue
        existing = conn.execute(
            "SELECT id, deleted_at, is_active, role, semester FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            if existing['role'] != 'student':
                skipped.append({'username': username, 'reason': '非學生帳號，無法匯入'})
                continue
            sync_enrollment(conn, existing['id'])
            if existing["deleted_at"] is not None or not existing['is_active']:
                # Re-enrolling student: restore the archived account so their
                # learning history (keyed by user id) stays linked.
                initial_password = secrets.token_urlsafe(9)  # 12 chars
                conn.execute(
                    "UPDATE users SET deleted_at = NULL, is_active = 1, approval_status = 'approved', password_hash = ?, "
                    "must_change_password = 1, session_version = session_version + 1, semester = ?, updated_at = datetime('now') "
                    "WHERE id = ?",
                    (hash_password(initial_password), semester, existing["id"]),
                )
                restored.append({"username": username, "initial_password": initial_password})
                restored_ids.append(existing["id"])
                conn.execute('UPDATE users SET class_name = ? WHERE id = ?', (req.class_name.strip(), existing['id']))
                sync_enrollment(conn, existing['id'])
            else:
                skipped.append({"username": username, "reason": "帳號已存在"})
            continue
        initial_password = secrets.token_urlsafe(9)  # 12 chars
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, display_name, email, role, "
            "semester, must_change_password) VALUES (?, ?, ?, ?, 'student', ?, 1)",
            (username, hash_password(initial_password),
             row.display_name.strip() or username, row.email.strip(), semester),
        )
        conn.execute('UPDATE users SET class_name = ? WHERE id = ?', (req.class_name.strip(), cursor.lastrowid))
        sync_enrollment(conn, cursor.lastrowid)
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


def _set_approval(user_id: int, approve: bool, request: Request, admin: dict) -> UserOut:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ? AND deleted_at IS NULL", (user_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="使用者不存在")
    if row["approval_status"] != "pending":
        conn.close()
        raise HTTPException(status_code=400, detail="此帳號不是待審核狀態")
    conn.execute(
        "UPDATE users SET approval_status = ?, is_active = ?, session_version = session_version + 1, "
        "updated_at = datetime('now') WHERE id = ?",
        ("approved" if approve else "rejected", int(approve), user_id),
    )
    if approve:
        sync_enrollment(conn, user_id)
    conn.commit()
    updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    log_audit("user.signup_approve" if approve else "user.signup_reject", actor=admin,
              target_type="user", target_id=user_id, detail={"username": row["username"]}, ip=client_ip(request))
    return _user_out(updated)


@router.post("/users/{user_id}/approve", response_model=UserOut)
async def approve_signup(user_id: int, request: Request, admin: dict = Depends(require_admin)):
    return _set_approval(user_id, True, request, admin)


@router.post("/users/{user_id}/reject", response_model=UserOut)
async def reject_signup(user_id: int, request: Request, admin: dict = Depends(require_admin)):
    return _set_approval(user_id, False, request, admin)


@router.post("/users/bulk")
async def bulk_user_action(req: BulkUserAction, request: Request, admin: dict = Depends(require_admin)):
    """Apply one action to many accounts. The caller's own account is always skipped
    so an admin cannot lock themselves out."""
    ids = sorted({i for i in req.ids if i != admin["id"]})
    if req.action == "set_class" and not req.class_name.strip():
        raise HTTPException(status_code=400, detail="請輸入班級名稱")
    conn = get_db()
    placeholders = ",".join("?" * len(ids)) or "NULL"
    rows = conn.execute(
        f"SELECT id, username FROM users WHERE id IN ({placeholders}) AND deleted_at IS NULL "
        "AND approval_status = 'approved'",
        ids,
    ).fetchall()
    passwords = []
    for row in rows:
        if req.action == "activate":
            conn.execute("UPDATE users SET is_active = 1, session_version = session_version + 1, "
                         "updated_at = datetime('now') WHERE id = ?", (row["id"],))
        elif req.action == "deactivate":
            conn.execute("UPDATE users SET is_active = 0, session_version = session_version + 1, "
                         "updated_at = datetime('now') WHERE id = ?", (row["id"],))
        elif req.action == "reset_password":
            pw = secrets.token_urlsafe(9)  # 12 chars
            conn.execute("UPDATE users SET password_hash = ?, must_change_password = 1, "
                         "session_version = session_version + 1, updated_at = datetime('now') WHERE id = ?",
                         (hash_password(pw), row["id"]))
            passwords.append({"username": row["username"], "initial_password": pw})
        elif req.action == "set_class":
            conn.execute("UPDATE users SET class_name = ?, updated_at = datetime('now') WHERE id = ?",
                         (req.class_name.strip(), row["id"]))
        sync_enrollment(conn, row["id"])
    conn.commit()
    conn.close()
    log_audit(f"user.bulk_{req.action}", actor=admin, target_type="user",
              detail={"ids": [r["id"] for r in rows]}, ip=client_ip(request))
    return {"updated": len(rows), "passwords": passwords}


@router.post("/semesters/{semester}/archive")
async def archive_semester(semester: str, request: Request, admin: dict = Depends(require_admin)):
    conn = get_db()
    cursor = conn.execute(
        "UPDATE users SET is_active = 0, session_version = session_version + 1, updated_at = datetime('now') "
        "WHERE semester = ? AND role = 'student' AND is_active = 1 AND deleted_at IS NULL",
        (semester,),
    )
    conn.commit()
    conn.close()
    ip = client_ip(request)
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
    ip = client_ip(request)
    log_audit("teacher_student.assign", actor=admin, target_type="teacher_student",
              target_id=f"{teacher_id}:{student_id}", ip=ip)
    return {"status": "assigned"}


@router.delete("/teachers/{teacher_id}/students/{student_id}")
async def remove_student_from_teacher(teacher_id: int, student_id: int, request: Request, admin: dict = Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM teacher_students WHERE teacher_id = ? AND student_id = ?", (teacher_id, student_id))
    conn.commit()
    conn.close()
    ip = client_ip(request)
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
    ip = client_ip(request)
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

        ip = client_ip(request)
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
    ip = client_ip(request)
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
        raise HTTPException(400, "缺少必要欄位：" + "、".join(sorted(required - body.keys())))
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
    ip = client_ip(request)
    log_audit("quiz.create", actor=_user, target_type="quiz_question", target_id=body["id"], ip=ip)
    return {"question": q}


@router.put("/quiz/questions/{question_id}")
async def admin_update_question(question_id: str, body: dict, request: Request, _user=Depends(require_admin)):
    from app.quiz.questions import update_question
    q = update_question(question_id, body)
    if not q:
        raise HTTPException(404, "題目不存在")
    ip = client_ip(request)
    log_audit("quiz.update", actor=_user, target_type="quiz_question", target_id=question_id, ip=ip)
    return {"question": q}


@router.delete("/quiz/questions/{question_id}")
async def admin_delete_question(question_id: str, request: Request, _user=Depends(require_admin)):
    from app.quiz.questions import delete_question
    if not delete_question(question_id):
        raise HTTPException(404, "題目不存在")
    ip = client_ip(request)
    log_audit("quiz.delete", actor=_user, target_type="quiz_question", target_id=question_id, ip=ip)
    return {"deleted": True}

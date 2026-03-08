from fastapi import APIRouter, HTTPException, Depends, Query
from app.auth.models import UserOut, UserUpdate, UserCreate
from app.auth.utils import hash_password
from app.auth.dependencies import require_admin, require_teacher_or_admin
from app.db import get_db, get_all_settings, set_setting
from app.llm.factory import list_available_providers
from app.nlp.trainer import train_models

router = APIRouter(prefix="/api/admin", tags=["Admin"])


def _user_out(row) -> UserOut:
    return UserOut(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        email=row["email"],
        role=row["role"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ── User CRUD (admin only) ──


@router.get("/users", response_model=list[UserOut])
async def list_users(
    role: str | None = Query(None),
    admin: dict = Depends(require_admin),
):
    conn = get_db()
    if role:
        rows = conn.execute("SELECT * FROM users WHERE role = ? ORDER BY id", (role,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()
    return [_user_out(r) for r in rows]


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int, admin: dict = Depends(require_admin)):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="使用者不存在")
    return _user_out(row)


@router.put("/users/{user_id}", response_model=UserOut)
async def update_user(user_id: int, data: UserUpdate, admin: dict = Depends(require_admin)):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
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
    if data.password is not None:
        updates.append("password_hash = ?")
        params.append(hash_password(data.password))

    if updates:
        updates.append("updated_at = datetime('now')")
        params.append(user_id)
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

    updated = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return _user_out(updated)


@router.delete("/users/{user_id}")
async def deactivate_user(user_id: int, admin: dict = Depends(require_admin)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="無法停用自己的帳號")
    conn = get_db()
    conn.execute("UPDATE users SET is_active = 0, updated_at = datetime('now') WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "deactivated"}


# ── Teacher-Student assignment (admin only) ──


@router.post("/teachers/{teacher_id}/students/{student_id}")
async def assign_student_to_teacher(teacher_id: int, student_id: int, admin: dict = Depends(require_admin)):
    conn = get_db()
    teacher = conn.execute("SELECT * FROM users WHERE id = ? AND role = 'teacher'", (teacher_id,)).fetchone()
    student = conn.execute("SELECT * FROM users WHERE id = ? AND role = 'student'", (student_id,)).fetchone()
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
    return {"status": "assigned"}


@router.delete("/teachers/{teacher_id}/students/{student_id}")
async def remove_student_from_teacher(teacher_id: int, student_id: int, admin: dict = Depends(require_admin)):
    conn = get_db()
    conn.execute("DELETE FROM teacher_students WHERE teacher_id = ? AND student_id = ?", (teacher_id, student_id))
    conn.commit()
    conn.close()
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
           WHERE ts.teacher_id = ? ORDER BY u.id""",
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
async def update_settings(data: dict, admin: dict = Depends(require_admin)):
    """Update system settings. Accepts key-value pairs."""
    allowed_keys = {"llm_provider", "llm_model", "rag_enabled", "rag_top_k"}
    for key, value in data.items():
        if key not in allowed_keys:
            raise HTTPException(status_code=400, detail=f"不允許的設定項: {key}")
        set_setting(key, str(value))
    return {"status": "updated", "settings": get_all_settings()}


# ── NLP Model Training (admin only) ──


@router.post("/train-nlp")
async def train_nlp_models(admin: dict = Depends(require_admin)):
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

        return {"status": "trained", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"訓練失敗: {str(e)}")

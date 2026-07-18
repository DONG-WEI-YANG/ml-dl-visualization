from fastapi import APIRouter, HTTPException, status, Depends, Request
from app.auth.models import LoginRequest, TokenResponse, UserOut, UserCreate, ChangePasswordRequest
from app.auth.utils import verify_password, hash_password, create_token
from app.auth.dependencies import get_current_user, require_admin
from app.db import get_db, get_setting
from app.audit import log_audit

router = APIRouter(prefix="/api/auth", tags=["Auth"])


def _user_out(row: dict) -> UserOut:
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


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request):
    ip = request.client.host if request.client else ""
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? AND deleted_at IS NULL", (req.username,)
    ).fetchone()
    conn.close()
    if not user or not verify_password(req.password, user["password_hash"]):
        log_audit("login.failed", detail={"username": req.username}, ip=ip)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="帳號或密碼錯誤")
    if not user["is_active"]:
        log_audit("login.failed", detail={"username": req.username, "reason": "inactive"}, ip=ip)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="帳號已停用")
    token = create_token(user["id"], user["username"], user["role"])
    log_audit("login.success", actor=dict(user), ip=ip)
    return TokenResponse(access_token=token, user=_user_out(dict(user)))


@router.post("/logout")
async def logout(request: Request, user: dict = Depends(get_current_user)):
    ip = request.client.host if request.client else ""
    log_audit("logout", actor=user, ip=ip)
    return {"status": "ok"}


@router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)):
    return _user_out(user)


@router.post("/change-password")
async def change_password(
    req: ChangePasswordRequest, request: Request, user: dict = Depends(get_current_user)
):
    ip = request.client.host if request.client else ""
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="新密碼長度至少 8 碼")
    if not verify_password(req.old_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="舊密碼錯誤")
    conn = get_db()
    conn.execute(
        "UPDATE users SET password_hash = ?, must_change_password = 0, "
        "updated_at = datetime('now') WHERE id = ?",
        (hash_password(req.new_password), user["id"]),
    )
    conn.commit()
    conn.close()
    log_audit("user.password_change", actor=user, target_type="user", target_id=user["id"], ip=ip)
    return {"status": "ok"}


@router.post("/register", response_model=UserOut)
async def register(req: UserCreate, admin: dict = Depends(require_admin)):
    """Admin creates new user accounts."""
    if req.role not in ("admin", "teacher", "student"):
        raise HTTPException(status_code=400, detail="角色必須為 admin、teacher 或 student")
    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (req.username,)).fetchone()
    if existing:
        raise HTTPException(status_code=409, detail="使用者名稱已存在")
    semester = req.semester or get_setting("current_semester", "")
    cursor = conn.execute(
        "INSERT INTO users (username, password_hash, display_name, email, role, semester) VALUES (?, ?, ?, ?, ?, ?)",
        (req.username, hash_password(req.password), req.display_name or req.username, req.email, req.role, semester),
    )
    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
    conn.close()
    return _user_out(dict(user))

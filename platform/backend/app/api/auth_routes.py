from fastapi import APIRouter, HTTPException, status, Depends
from app.auth.models import LoginRequest, TokenResponse, UserOut, UserCreate
from app.auth.utils import verify_password, hash_password, create_token
from app.auth.dependencies import get_current_user, require_admin
from app.db import get_db, get_setting

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
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (req.username,)).fetchone()
    conn.close()
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="帳號或密碼錯誤")
    if not user["is_active"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="帳號已停用")
    token = create_token(user["id"], user["username"], user["role"])
    return TokenResponse(access_token=token, user=_user_out(dict(user)))


@router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)):
    return _user_out(user)


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

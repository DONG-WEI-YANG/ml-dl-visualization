from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.auth.utils import decode_token
from app.db import get_db

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="無效或過期的令牌")
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ? AND is_active = 1", (payload["sub"],)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="使用者不存在或已停用")
    return dict(user)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理員權限")
    return user


def require_teacher_or_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] not in ("admin", "teacher"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要教師或管理員權限")
    return user

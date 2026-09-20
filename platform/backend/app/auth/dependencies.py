from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.auth.utils import decode_token
from app.db import db_connection

security = HTTPBearer()


def authenticate_token(token: str, *, allow_password_change: bool = False) -> dict:
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="無效或過期的令牌")
    with db_connection() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE id = ? AND is_active = 1 AND deleted_at IS NULL",
            (payload["sub"],),
        ).fetchone()
        revoked = conn.execute('SELECT 1 FROM revoked_tokens WHERE jti = ?', (payload['jti'],)).fetchone()
    if not user or revoked or user['session_version'] != payload['ver']:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="使用者不存在或已停用")
    if user['must_change_password'] and not allow_password_change:
        raise HTTPException(status_code=403, detail='請先變更初始密碼')
    return dict(user)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    return authenticate_token(credentials.credentials)


def get_password_change_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    return authenticate_token(credentials.credentials, allow_password_change=True)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理員權限")
    return user


def require_teacher_or_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] not in ("admin", "teacher"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要教師或管理員權限")
    return user

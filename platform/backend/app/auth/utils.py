import hashlib
import hmac
import os
import json
import base64
import time
from app.config import settings


def hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return (salt + key).hex()


def verify_password(password: str, stored_hash: str) -> bool:
    data = bytes.fromhex(stored_hash)
    salt, key = data[:32], data[32:]
    new_key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return hmac.compare_digest(key, new_key)


def create_token(user_id: int, username: str, role: str) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=")
    payload_data = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": int(time.time()) + settings.jwt_expire_minutes * 60,
    }
    payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).rstrip(b"=")
    signing_input = header + b"." + payload
    signature = base64.urlsafe_b64encode(
        hmac.new(settings.jwt_secret.encode(), signing_input, hashlib.sha256).digest()
    ).rstrip(b"=")
    return (signing_input + b"." + signature).decode()


def decode_token(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        signing_input = (parts[0] + "." + parts[1]).encode()
        signature = base64.urlsafe_b64decode(parts[2] + "==")
        expected = hmac.new(settings.jwt_secret.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None

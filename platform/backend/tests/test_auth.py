from app.auth.utils import hash_password, verify_password, create_token, decode_token


def test_password_hash_verify():
    pw = "mypassword123"
    h = hash_password(pw)
    assert verify_password(pw, h)
    assert not verify_password("wrongpassword", h)


def test_token_create_decode():
    token = create_token(1, "admin", "admin")
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "1"
    assert payload["username"] == "admin"
    assert payload["role"] == "admin"


def test_token_invalid():
    assert decode_token("invalid.token.here") is None
    assert decode_token("") is None

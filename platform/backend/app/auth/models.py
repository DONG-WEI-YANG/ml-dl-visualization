from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    password: str
    display_name: str = ""
    email: str = ""
    semester: str = ""
    role: str = "student"


class UserUpdate(BaseModel):
    display_name: str | None = None
    email: str | None = None
    role: str | None = None
    is_active: bool | None = None
    semester: str | None = None
    password: str | None = None


class UserOut(BaseModel):
    id: int
    username: str
    display_name: str
    email: str
    semester: str
    role: str
    is_active: bool
    created_at: str
    updated_at: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

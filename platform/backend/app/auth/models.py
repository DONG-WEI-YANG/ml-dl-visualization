from typing import Literal

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    class_name: str = ''
    username: str
    password: str
    display_name: str = ""
    email: str = ""
    semester: str = ""
    role: str = "student"


class UserUpdate(BaseModel):
    class_name: str | None = None
    display_name: str | None = None
    email: str | None = None
    role: str | None = None
    is_active: bool | None = None
    semester: str | None = None
    password: str | None = None


class UserOut(BaseModel):
    class_name: str = ''
    id: int
    username: str
    display_name: str
    email: str
    semester: str
    role: str
    is_active: bool
    created_at: str
    updated_at: str
    must_change_password: bool = False
    approval_status: str = 'approved'


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class ImportRow(BaseModel):
    username: str
    display_name: str = ""
    email: str = ""


class ImportRequest(BaseModel):
    class_name: str = ''
    semester: str = ""
    rows: list[ImportRow]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class SignupRequest(BaseModel):
    """Public self-service sign-up; always creates a pending student."""
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=50)
    class_name: str = Field(default="", max_length=50)
    email: str = Field(default="", max_length=120)


class BulkUserAction(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)
    action: Literal["activate", "deactivate", "reset_password", "set_class"]
    class_name: str = ""

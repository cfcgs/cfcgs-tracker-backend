from pydantic import BaseModel, ConfigDict, EmailStr

from src.cfcgs_tracker.domain.models import UserRole


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.importer


class UserUpdate(BaseModel):
    username: str
    email: EmailStr
    current_password: str | None = None
    new_password: str | None = None
    new_password_confirmation: str | None = None


class UserRoleUpdate(BaseModel):
    role: UserRole


class PasswordVerificationRequest(BaseModel):
    current_password: str


class UserPublic(BaseModel):
    username: str
    email: EmailStr
    id: int
    role: UserRole
    is_active: bool
    model_config = ConfigDict(from_attributes=True)


class UserList(BaseModel):
    users: list[UserPublic]

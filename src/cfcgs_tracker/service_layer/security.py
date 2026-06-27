from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from jwt import encode
from pwdlib import PasswordHash

from src.cfcgs_tracker.settings import Settings

pwd_context = PasswordHash.recommended()


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    return create_token(
        data=data,
        expires_delta=timedelta(
            minutes=Settings().ACCESS_TOKEN_EXPIRE_MINUTES
        ),
        token_type="access",
    )


def create_refresh_token(data: dict) -> str:
    return create_token(
        data=data,
        expires_delta=timedelta(days=Settings().REFRESH_TOKEN_EXPIRE_DAYS),
        token_type="refresh",
    )


def create_token(
    *,
    data: dict,
    expires_delta: timedelta,
    token_type: str,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(tz=ZoneInfo("UTC")) + expires_delta
    to_encode.update({"exp": expire, "token_type": token_type})

    return encode(
        to_encode, Settings().SECRET_KEY, algorithm=Settings().ALGORITHM
    )

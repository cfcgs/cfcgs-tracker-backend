from typing import Set
from pydantic_settings import BaseSettings, SettingsConfigDict


# Função para transformar string CSV em set
def comma_split_str(value: str | set[str]) -> set[str]:
    if isinstance(value, str):
        return set(s.strip() for s in value.split(","))
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    DATABASE_URL: str
    GEMINI_API_KEY: str
    CHATBOT_RATE_LIMIT_REQUESTS: int = 30
    CHATBOT_RATE_LIMIT_WINDOW_SECONDS: int = 60
    CHATBOT_RATE_LIMIT_ENABLED: bool = True
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    INITIAL_ADMIN_USERNAME: str | None = None
    INITIAL_ADMIN_EMAIL: str | None = None
    INITIAL_ADMIN_PASSWORD: str | None = None
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
    EXPECTED_COLUMNS: str
    SIMILARITY_THRESHOLD: int
    REGIONS: str

    @property
    def expected_columns_set(self) -> Set[str]:
        return comma_split_str(self.EXPECTED_COLUMNS)

    @property
    def regions_set(self) -> Set[str]:
        return comma_split_str(self.REGIONS)

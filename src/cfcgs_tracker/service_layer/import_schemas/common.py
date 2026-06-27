import re
from typing import Any

from pydantic import BaseModel, ConfigDict


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.lower() in {"not applicable", "n/a", "na", "none", "null"}:
        return None

    return text


def normalize_float(value: Any) -> float | None:
    normalized = normalize_text(value)
    if normalized is None:
        return None

    compact = re.sub(r"[\s$,]", "", normalized)
    return float(compact)


def normalize_int(value: Any) -> int | None:
    normalized = normalize_text(value)
    if normalized is None:
        return None

    compact = re.sub(r"[\s$,]", "", normalized)
    return int(float(compact))


class ImportRowSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

import os
from pathlib import Path
from urllib.parse import urlsplit
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

if os.getenv("API_LOAD_DOTENV", "true").lower() not in ("false", "0", "no", "off"):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (true/false).")


def llm_base_url() -> str:
    value = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("LLM_BASE_URL must be an HTTP(S) API base URL without a query or fragment.")
    parsed.port  # Reject invalid port numbers before the first translation.
    return value


def parse_origins(value: Optional[str]) -> List[str]:
    """Parse comma-separated CORS origins while discarding empty entries."""
    return [origin.strip() for origin in (value or "").split(",") if origin.strip()]


def validate_cors_configuration(origins: List[str], allow_credentials: bool) -> None:
    """Reject the insecure wildcard-and-credentials CORS combination."""
    if allow_credentials and "*" in origins:
        raise ValueError("ALLOWED_ORIGINS cannot contain '*' when CORS_ALLOW_CREDENTIALS is enabled.")


class Settings(BaseModel):
    PROJECT_NAME: str = "Manga Translator API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    HOST: str = os.getenv("API_HOST", "127.0.0.1")
    PORT: int = Field(default=int(os.getenv("API_PORT", "8000")), ge=1, le=65535)
    DEBUG: bool = env_bool("API_DEBUG")
    RELOAD: bool = env_bool("API_RELOAD", DEBUG)

    ALLOWED_ORIGINS: List[str] = parse_origins(
        os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )
    CORS_ALLOW_CREDENTIALS: bool = env_bool("CORS_ALLOW_CREDENTIALS")

    MAX_UPLOAD_BYTES: int = Field(default=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))), gt=0)
    MAX_IMAGE_PIXELS: int = Field(default=int(os.getenv("MAX_IMAGE_PIXELS", "40000000")), gt=0)
    MAX_IMAGE_REQUESTS: int = Field(default=int(os.getenv("MAX_IMAGE_REQUESTS", "4")), gt=0)

    SUPPORTED_SOURCE_LANGS: List[str] = ["ja", "ko", "zh", "en"]
    SUPPORTED_TARGET_LANGS: List[str] = ["id", "en"]

    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = llm_base_url()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT_SECONDS: float = Field(default=float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0")), gt=0, allow_inf_nan=False)
    DEFAULT_TRANSLATOR: Literal["google", "llm"] = os.getenv("DEFAULT_TRANSLATOR", "llm")

    model_config = {"validate_default": True}


settings = Settings()
validate_cors_configuration(settings.ALLOWED_ORIGINS, settings.CORS_ALLOW_CREDENTIALS)

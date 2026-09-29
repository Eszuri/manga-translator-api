import os
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


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
    
    # Server network settings
    HOST: str = os.getenv("API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("API_DEBUG", "False").lower() in ("true", "1")

    # CORS settings. Add exact browser-extension origins through ALLOWED_ORIGINS.
    ALLOWED_ORIGINS: List[str] = parse_origins(
        os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )
    CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "False").lower() in ("true", "1")

    # Upload limits protect memory usage and decompression work for image endpoints.
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))

    # Supported language codes (ISO-639-1)
    SUPPORTED_SOURCE_LANGS: List[str] = ["ja", "ko", "zh", "en"]
    SUPPORTED_TARGET_LANGS: List[str] = ["id", "en"]

    # Translation Engine (OpenAI-Compatible: OpenAI, Groq, OpenRouter, DeepSeek, Ollama, etc.)
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))


settings = Settings()
validate_cors_configuration(settings.ALLOWED_ORIGINS, settings.CORS_ALLOW_CREDENTIALS)

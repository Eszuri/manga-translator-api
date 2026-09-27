import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    PROJECT_NAME: str = "Manga Translator API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Server network settings
    HOST: str = os.getenv("API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("API_DEBUG", "True").lower() in ("true", "1")

    # CORS settings - allow browser extensions and local frontend readers
    ALLOWED_ORIGINS: List[str] = [
        "*",  # Needed for chrome-extension:// and moz-extension://
    ]

    # Supported language codes (ISO-639-1)
    SUPPORTED_SOURCE_LANGS: List[str] = ["ja", "ko", "zh", "en"]
    SUPPORTED_TARGET_LANGS: List[str] = ["id", "en"]

    # Translation Engine (OpenAI-Compatible: OpenAI, Groq, OpenRouter, DeepSeek, Ollama, etc.)
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    DEFAULT_TARGET_LANG: str = os.getenv("DEFAULT_TARGET_LANG", "id")
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))


settings = Settings()

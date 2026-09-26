from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    project_name: str
    version: str
    server_time_utc: str
    supported_source_languages: List[str]
    supported_target_languages: List[str]


@router.get("/health", response_model=HealthResponse)
def get_health():
    """
    Health check endpoint to verify API server is operational
    and retrieve supported translation languages.
    """
    return HealthResponse(
        status="healthy",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        server_time_utc=datetime.now(timezone.utc).isoformat(),
        supported_source_languages=settings.SUPPORTED_SOURCE_LANGS,
        supported_target_languages=settings.SUPPORTED_TARGET_LANGS
    )

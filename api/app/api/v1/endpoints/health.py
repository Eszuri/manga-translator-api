from datetime import datetime, timezone
from typing import List
import logging
import time
from collections import deque
from fastapi import APIRouter
from pydantic import BaseModel, Field
from fastapi import HTTPException
from app.core.config import settings
from app.core.gpu import required_gpu_provider

router = APIRouter()
logger = logging.getLogger(__name__)
_client_error_times = deque()


class ExtensionError(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    source: str = Field(default="", max_length=2000)
    job_id: str = Field(default="", max_length=128)


@router.post("/client-error")
async def report_extension_error(error: ExtensionError):
    now = time.monotonic()
    while _client_error_times and now - _client_error_times[0] >= 60:
        _client_error_times.popleft()
    if len(_client_error_times) >= 60:
        raise HTTPException(status_code=429, detail="Client error log rate limit reached.")
    _client_error_times.append(now)
    clean = lambda value: " ".join(value.split())
    logger.error("[Extension] job=%s source=%s: %s", clean(error.job_id), clean(error.source), clean(error.message))
    return {"logged": True}


class HealthResponse(BaseModel):
    status: str
    project_name: str
    version: str
    server_time_utc: str
    supported_source_languages: List[str]
    supported_target_languages: List[str]
    compute_mode: str
    gpu_provider: str


@router.get("/health", response_model=HealthResponse)
def get_health():
    return HealthResponse(
        status="healthy",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        server_time_utc=datetime.now(timezone.utc).isoformat(),
        supported_source_languages=settings.SUPPORTED_SOURCE_LANGS,
        supported_target_languages=settings.SUPPORTED_TARGET_LANGS,
        compute_mode="gpu-required-no-runtime-fallback",
        gpu_provider=required_gpu_provider(),
    )

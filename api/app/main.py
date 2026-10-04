from contextlib import asynccontextmanager
import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.gpu import required_gpu_provider
from app.api.v1.router import api_router
from app.services.translation_service import TranslationError

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail startup unless both inference models create strict GPU sessions."""
    startup_started = perf_counter()
    from app.services.comic_text_detector import ComicTextDetector
    from app.services.ocr_service import get_ocr_service

    app.state.gpu_provider = required_gpu_provider()
    app.state.detector = ComicTextDetector(device="gpu")
    app.state.ocr = get_ocr_service(device="gpu")
    logger.info(
        "[startup:ready] GPU models ready in %.1fs; starting HTTP server",
        perf_counter() - startup_started,
        extra={"startup_phase": "models_ready"},
    )
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for Manga Translator Browser Extension (Speech Bubble Detection, Manga-OCR, & Translation).",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log rejected form fields without logging uploaded image data."""
    errors = [
        {key: value for key, value in error.items() if key in ("type", "loc", "msg")}
        for error in exc.errors()
    ]
    logger.warning(
        "Request validation rejected %s %s: %s",
        request.method,
        request.url.path,
        errors,
    )
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


@app.exception_handler(TranslationError)
async def translation_error(request: Request, exc: TranslationError) -> JSONResponse:
    logger.warning("Translation failed on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/")
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health_check": f"{settings.API_V1_PREFIX}/health"
    }

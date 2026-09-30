from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.gpu import required_gpu_provider
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail startup unless both inference models create strict GPU sessions."""
    from app.services.comic_text_detector import ComicTextDetector
    from app.services.ocr_service import get_ocr_service

    app.state.gpu_provider = required_gpu_provider()
    app.state.detector = ComicTextDetector(device="gpu")
    app.state.ocr = get_ocr_service(device="gpu")
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


@app.get("/")
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health_check": f"{settings.API_V1_PREFIX}/health"
    }

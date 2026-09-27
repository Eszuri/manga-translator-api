from fastapi import APIRouter
from app.api.v1.endpoints import health, detect, ocr

api_router = APIRouter()

# Core system endpoints
api_router.include_router(health.router, tags=["Health & Status"])

# Bubble & Text Detection endpoints
api_router.include_router(detect.router, prefix="/detect", tags=["Detection"])

# Manga OCR & Text Extraction endpoints
api_router.include_router(ocr.router, prefix="/ocr", tags=["OCR & Recognition"])


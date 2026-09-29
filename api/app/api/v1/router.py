from fastapi import APIRouter
from app.api.v1.endpoints import health, detect, ocr, translate

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Status"])
api_router.include_router(detect.router, prefix="/detect", tags=["Detection"])
api_router.include_router(ocr.router, prefix="/ocr", tags=["OCR & Recognition"])
api_router.include_router(translate.router, prefix="/translate", tags=["Translation"])


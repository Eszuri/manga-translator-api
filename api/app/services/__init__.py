"""Services package for Manga Translator"""
from app.services.detector import (
    BaseBubbleDetector,
    ContourBubbleDetector,
    sort_manga_reading_order,
    calculate_iou
)
from app.services.comic_text_detector import ComicTextDetector
from app.services.hybrid_detector import HybridBubbleDetector
from app.services.ocr_service import MangaOcrService, get_ocr_service

__all__ = [
    "BaseBubbleDetector",
    "ContourBubbleDetector",
    "ComicTextDetector",
    "HybridBubbleDetector",
    "MangaOcrService",
    "get_ocr_service",
    "sort_manga_reading_order",
    "calculate_iou"
]

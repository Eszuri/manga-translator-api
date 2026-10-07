import time
from typing import Literal
from fastapi import APIRouter, UploadFile, File, Form

from app.api.v1.image_uploads import read_validated_image
from app.core.image_worker import run_image_task
from app.schemas import DetectBubblesResponse
from app.services.detector import sort_manga_reading_order

router = APIRouter()

_hybrid_detector = None


def get_hybrid_detector():
    global _hybrid_detector
    if _hybrid_detector is None:
        from app.services.hybrid_detector import HybridBubbleDetector
        _hybrid_detector = HybridBubbleDetector()
    return _hybrid_detector


def detect_page(image, reading_direction):
    detector = get_hybrid_detector()
    return sort_manga_reading_order(detector.detect(image), reading_direction=reading_direction)


@router.post("/bubbles", response_model=DetectBubblesResponse)
async def detect_bubbles(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl", 
        description="Reading direction: 'rtl' (Japanese Manga) or 'ltr' (Korean Manhwa / Webtoon)"
    )
):
    image = await read_validated_image(file)
    start_time = time.perf_counter()
    ordered_bubbles = await run_image_task(detect_page, image, reading_direction)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return DetectBubblesResponse(
        success=True,
        image_width=image.width,
        image_height=image.height,
        total_detected=len(ordered_bubbles),
        reading_direction=reading_direction,
        bubbles=ordered_bubbles,
        processing_time_ms=duration_ms
    )

import time
from typing import Literal
from fastapi import APIRouter, UploadFile, File, Form

from app.api.v1.image_uploads import read_validated_image
from app.schemas import DetectBubblesResponse
from app.services.detector import sort_manga_reading_order

router = APIRouter()

_comic_text_detector = None
_hybrid_detector = None


def get_comic_text_detector():
    global _comic_text_detector
    if _comic_text_detector is None:
        from app.services.comic_text_detector import ComicTextDetector
        _comic_text_detector = ComicTextDetector(device="gpu", require_gpu=True)
    return _comic_text_detector


def get_hybrid_detector():
    global _hybrid_detector
    if _hybrid_detector is None:
        from app.services.hybrid_detector import HybridBubbleDetector
        _hybrid_detector = HybridBubbleDetector(device="gpu", require_gpu=True)
    return _hybrid_detector


@router.post("/bubbles", response_model=DetectBubblesResponse)
async def detect_bubbles(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl", 
        description="Reading direction: 'rtl' (Japanese Manga) or 'ltr' (Korean Manhwa / Webtoon)"
    ),
    detector_type: Literal["hybrid", "comic_text_detector"] = Form(
        "hybrid",
        description="GPU-backed detector: 'hybrid' or 'comic_text_detector'"
    )
):
    """
    Detect speech bubbles and text regions on a manga page.
    Returns a list of bounding boxes sorted according to comic reading order.
    """
    image = await read_validated_image(file)

    start_time = time.perf_counter()

    if detector_type == "comic_text_detector":
        active_detector = get_comic_text_detector()
    elif detector_type == "hybrid":
        active_detector = get_hybrid_detector()
    else:
        raise ValueError("GPU-only backend does not support the CPU contour detector.")

    detected = active_detector.detect(image)
    ordered_bubbles = sort_manga_reading_order(detected, reading_direction=reading_direction)

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

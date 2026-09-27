import io
import time
from typing import Literal
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from PIL import Image

from app.schemas import DetectBubblesResponse
from app.services.detector import ContourBubbleDetector, sort_manga_reading_order

router = APIRouter()

contour_detector = ContourBubbleDetector()
_comic_text_detector = None
_hybrid_detector = None


def get_comic_text_detector():
    global _comic_text_detector
    if _comic_text_detector is None:
        from app.services.comic_text_detector import ComicTextDetector
        _comic_text_detector = ComicTextDetector()
    return _comic_text_detector


def get_hybrid_detector():
    global _hybrid_detector
    if _hybrid_detector is None:
        from app.services.hybrid_detector import HybridBubbleDetector
        _hybrid_detector = HybridBubbleDetector()
    return _hybrid_detector


@router.post("/bubbles", response_model=DetectBubblesResponse)
async def detect_bubbles(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl", 
        description="Reading direction: 'rtl' (Japanese Manga) or 'ltr' (Korean Manhwa / Webtoon)"
    ),
    detector_type: Literal["hybrid", "comic_text_detector", "contour"] = Form(
        "hybrid",
        description="Detector engine: 'hybrid' (AI + OpenCV Balloon - Recommended), 'comic_text_detector' (Deep Learning AI), or 'contour' (OpenCV)"
    )
):
    """
    Detect speech bubbles and text regions on a manga page.
    Returns a list of bounding boxes sorted according to comic reading order.
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp", "application/octet-stream"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File format '{file.content_type}' is not supported. Please use JPEG, PNG, or WebP."
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty."
            )
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image file: {str(e)}"
        )

    start_time = time.perf_counter()

    # 1. Execute speech bubble & text detection
    if detector_type == "comic_text_detector":
        active_detector = get_comic_text_detector()
    elif detector_type == "hybrid":
        active_detector = get_hybrid_detector()
    else:
        active_detector = contour_detector

    detected = active_detector.detect(image)

    # 2. Sort according to comic reading order (RTL / LTR)
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

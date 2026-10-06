import time
from typing import Literal
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.api.v1.image_uploads import read_validated_image
from app.schemas import DetectBubblesResponse
from app.core.image_worker import run_image_task
from app.services.detector import sort_manga_reading_order
from app.services.ocr_service import get_ocr_service

router = APIRouter()

_hybrid_detector = None


def get_hybrid_detector():
    global _hybrid_detector
    if _hybrid_detector is None:
        from app.services.hybrid_detector import HybridBubbleDetector
        from app.services.comic_text_detector import ComicTextDetector
        comic_det = ComicTextDetector(device="gpu")
        _hybrid_detector = HybridBubbleDetector(comic_detector=comic_det)
    return _hybrid_detector


def recognize_page(image, reading_direction):
    detector = get_hybrid_detector()
    bubbles = sort_manga_reading_order(detector.detect(image), reading_direction=reading_direction)
    return get_ocr_service(device="gpu").recognize_all_bubbles(image, bubbles)


def recognize_crop(image):
    return get_ocr_service(device="gpu").recognize_crop(image)


class CropOCRResponse(BaseModel):
    success: bool = True
    text: str = Field(..., description="Extracted Japanese dialogue text from Manga OCR")
    processing_time_ms: float


@router.post("/recognize", response_model=DetectBubblesResponse)
async def recognize_page_text(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
    )
):
    image = await read_validated_image(file)
    start_time = time.perf_counter()
    ordered_bubbles = await run_image_task(
        recognize_page, image, reading_direction
    )
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


@router.post("/recognize-crop", response_model=CropOCRResponse)
async def recognize_crop_text(
    file: UploadFile = File(..., description="Cropped text image (manual box selection from extension)")
):
    crop_img = await read_validated_image(file)
    start_time = time.perf_counter()
    text = await run_image_task(recognize_crop, crop_img)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return CropOCRResponse(
        success=True,
        text=text,
        processing_time_ms=duration_ms
    )

import time
from typing import Literal
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.api.v1.image_uploads import read_validated_image
from app.schemas import DetectBubblesResponse, DetectedBubble
from app.services.detector import ContourBubbleDetector, sort_manga_reading_order
from app.services.ocr_service import get_ocr_service

router = APIRouter()

_contour_detector = None
_comic_detector = None
_hybrid_detector = None


def get_detector_instance(detector_type: str, device: str):
    global _contour_detector, _comic_detector, _hybrid_detector
    if detector_type == "comic_text_detector":
        if _comic_detector is None:
            from app.services.comic_text_detector import ComicTextDetector
            _comic_detector = ComicTextDetector(device=device)
        return _comic_detector
    elif detector_type == "hybrid":
        if _hybrid_detector is None:
            from app.services.hybrid_detector import HybridBubbleDetector
            from app.services.comic_text_detector import ComicTextDetector
            comic_det = ComicTextDetector(device=device)
            _hybrid_detector = HybridBubbleDetector(comic_detector=comic_det)
        return _hybrid_detector
    else:
        if _contour_detector is None:
            _contour_detector = ContourBubbleDetector()
        return _contour_detector


class CropOCRResponse(BaseModel):
    success: bool = True
    text: str = Field(..., description="Extracted Japanese dialogue text from Manga OCR")
    processing_time_ms: float


@router.post("/recognize", response_model=DetectBubblesResponse)
async def recognize_page_text(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    detector_type: Literal["hybrid", "comic_text_detector", "contour"] = Form(
        "hybrid",
        description="Text/bubble detector engine ('hybrid' recommended for optimal accuracy)"
    ),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
    ),
    device: Literal["auto", "gpu", "cpu"] = Form(
        "auto",
        description="Compute device: 'gpu' (DirectML), 'cpu', or 'auto'"
    )
):
    """
    Full-page text extraction for a manga page:
    1. Automatically detects speech bubble and text areas.
    2. Sorts bubbles according to comic reading order (RTL/LTR).
    3. Crops each text region and extracts Japanese text using Manga-OCR (ViT).
    """
    image = await read_validated_image(file)

    start_time = time.perf_counter()

    detector = get_detector_instance(detector_type, device)
    detected = detector.detect(image)
    ordered_bubbles = sort_manga_reading_order(detected, reading_direction=reading_direction)

    ocr_service = get_ocr_service(device=device)
    ordered_bubbles = ocr_service.recognize_all_bubbles(image, ordered_bubbles)

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
    file: UploadFile = File(..., description="Cropped text image (manual box selection from extension)"),
    device: Literal["auto", "gpu", "cpu"] = Form(
        "auto",
        description="Compute device: 'gpu' (DirectML), 'cpu', or 'auto'"
    )
):
    """
    Extract text directly from a cropped image (extension manual box selection feature).
    """
    crop_img = await read_validated_image(file)

    start_time = time.perf_counter()
    ocr_service = get_ocr_service(device=device)
    text = ocr_service.recognize_crop(crop_img)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return CropOCRResponse(
        success=True,
        text=text,
        processing_time_ms=duration_ms
    )

import io
import time
from typing import Literal, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from PIL import Image

from app.core.config import settings
from app.schemas import (
    DetectBubblesResponse,
    DetectedBubble,
    TranslateDialoguesRequest,
    TranslateDialoguesResponse,
    TranslatedDialogueItem
)
from app.services.detector import ContourBubbleDetector, sort_manga_reading_order
from app.services.ocr_service import get_ocr_service
from app.services.translation_service import get_translation_service

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


@router.post("/page", response_model=DetectBubblesResponse)
async def translate_manga_page(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    target_lang: str = Form(
        "id",
        description="Target translation language code ('id' for Indonesian, 'en' for English)"
    ),
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
        description="Compute device for detection and OCR: 'gpu' (DirectML), 'cpu', or 'auto'"
    )
):
    """
    Complete End-to-End Manga Translation Pipeline:
    1. Speech Bubble & Text Detection (Hybrid AI + Balloon Segmentation).
    2. Reading Order Sorting (Right-to-Left / Left-to-Right).
    3. Japanese Text Recognition via Manga-OCR (ViT Transformer).
    4. Contextual Dialogue Translation via OpenAI-compatible LLM (default: Indonesian).
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

    # 1. Detection
    detector = get_detector_instance(detector_type, device)
    detected = detector.detect(image)

    # 2. Sort into reading order
    ordered_bubbles = sort_manga_reading_order(detected, reading_direction=reading_direction)

    # 3. OCR (Manga-OCR)
    ocr_service = get_ocr_service(device=device)
    ordered_bubbles = ocr_service.recognize_all_bubbles(image, ordered_bubbles)

    # 4. Contextual Translation (LLM)
    trans_service = get_translation_service()
    ordered_bubbles = await trans_service.translate_bubbles_async(ordered_bubbles, target_lang=target_lang)

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return DetectBubblesResponse(
        success=True,
        image_width=image.width,
        image_height=image.height,
        total_detected=len(ordered_bubbles),
        reading_direction=reading_direction,
        target_lang=target_lang,
        bubbles=ordered_bubbles,
        processing_time_ms=duration_ms
    )


@router.post("/dialogues", response_model=TranslateDialoguesResponse)
async def translate_dialogues(request: TranslateDialoguesRequest):
    """
    Direct contextual dialogue translation endpoint for client/extension.
    Translates an array of pre-extracted dialogues in reading order.
    """
    start_time = time.perf_counter()
    trans_service = get_translation_service()

    dialogue_dicts = [{"id": d.id, "text": d.text} for d in request.dialogues]
    translations_map = await trans_service._call_llm_async(dialogue_dicts, target_lang=request.target_lang)

    items = []
    for d in request.dialogues:
        items.append(
            TranslatedDialogueItem(
                id=d.id,
                original_text=d.text,
                translated_text=translations_map.get(d.id, d.text)
            )
        )

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return TranslateDialoguesResponse(
        success=True,
        target_lang=request.target_lang,
        dialogues=items,
        processing_time_ms=duration_ms,
        model=trans_service.model
    )

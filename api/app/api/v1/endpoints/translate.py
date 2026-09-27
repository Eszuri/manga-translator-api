import base64
import io
import time
from typing import Literal, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response, status
from PIL import Image

from app.core.config import settings
from app.schemas import (
    DetectBubblesResponse,
    DetectedBubble,
    InpaintPageResponse,
    TranslateDialoguesRequest,
    TranslateDialoguesResponse,
    TranslatedDialogueItem
)
from app.services.detector import ContourBubbleDetector, sort_manga_reading_order
from app.services.ocr_service import get_ocr_service
from app.services.translation_service import get_translation_service
from app.services.inpainting_service import MangaInpaintingService
from app.services.typesetting_service import MangaTypesettingService

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


@router.post("/inpaint-page")
async def inpaint_and_translate_manga_page(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    target_lang: str = Form(
        "id",
        description="Target translation language code ('id' for Indonesian, 'en' for English)"
    ),
    detector_type: Literal["hybrid", "comic_text_detector", "contour"] = Form(
        "hybrid",
        description="Text/bubble detector engine ('hybrid' recommended)"
    ),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
    ),
    device: Literal["auto", "gpu", "cpu"] = Form(
        "auto",
        description="Compute device for detection and OCR: 'gpu' (DirectML), 'cpu', or 'auto'"
    ),
    typeset: bool = Form(
        True,
        description="Whether to typeset translated text into bubbles (False for clean raw inpainting only)"
    ),
    font_scale: float = Form(
        1.0,
        description="Font size multiplier (default: 1.0)"
    ),
    all_caps: bool = Form(
        True,
        description="Render dialogue text in comic uppercase (default: True)"
    ),
    return_format: Literal["image", "json"] = Form(
        "image",
        description="Output format: 'image' (raw JPEG image binary) or 'json' (base64 image + bubble metadata)"
    )
):
    """
    End-to-End Server-Side Inpainting & Typesetting Pipeline:
    1. Speech Bubble & Text Detection.
    2. Character Segmentation Mask Retrieval.
    3. Japanese Dialogue Extraction (Manga-OCR).
    4. Contextual Dialogue Translation (LLM - default: Indonesian).
    5. Clean Text Inpainting / Erasure (Telea Inpainting on Dilated Character Mask).
    6. Comic Font Typesetting with dynamic auto-fit font sizing and balanced word wrapping.
    7. Returns rendered image binary stream or JSON with Base64 image and bubble details.
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
    ordered_bubbles = sort_manga_reading_order(detected, reading_direction=reading_direction)

    # 2. Extract neural segmentation mask if detector supports it
    seg_mask = None
    if detector_type in ("hybrid", "comic_text_detector"):
        comic_det = getattr(detector, "comic_detector", detector)
        if hasattr(comic_det, "detect_raw") and hasattr(comic_det, "get_unletterboxed_seg"):
            try:
                blk, seg, det, r, (dw, dh) = comic_det.detect_raw(image)
                seg_mask = comic_det.get_unletterboxed_seg(seg, image.width, image.height, dw, dh)
            except Exception:
                seg_mask = None

    # 3. OCR (Manga-OCR) if typesetting requested
    if typeset and ordered_bubbles:
        ocr_service = get_ocr_service(device=device)
        ordered_bubbles = ocr_service.recognize_all_bubbles(image, ordered_bubbles)

        # 4. Contextual Translation (LLM)
        trans_service = get_translation_service()
        ordered_bubbles = await trans_service.translate_bubbles_async(ordered_bubbles, target_lang=target_lang)

    # 5. Inpaint / Erase text
    inpaint_service = MangaInpaintingService()
    inpainted_img = inpaint_service.inpaint(image, seg_mask=seg_mask, bubbles=ordered_bubbles)

    # 6. Typeset translated text
    if typeset and ordered_bubbles:
        typeset_service = MangaTypesettingService(all_caps=all_caps)
        final_img = typeset_service.typeset(inpainted_img, ordered_bubbles, font_scale=font_scale)
    else:
        final_img = inpainted_img

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Ensure RGB mode for JPEG encoding
    if final_img.mode != "RGB":
        final_img = final_img.convert("RGB")

    out_buf = io.BytesIO()
    final_img.save(out_buf, format="JPEG", quality=95)
    img_bytes = out_buf.getvalue()

    if return_format == "image":
        return Response(
            content=img_bytes,
            media_type="image/jpeg",
            headers={
                "X-Processing-Time-Ms": str(duration_ms),
                "X-Total-Bubbles": str(len(ordered_bubbles)),
                "X-Target-Lang": target_lang
            }
        )

    # JSON format with base64 image
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return InpaintPageResponse(
        success=True,
        image_width=final_img.width,
        image_height=final_img.height,
        total_detected=len(ordered_bubbles),
        target_lang=target_lang if typeset else None,
        image_base64=img_b64,
        bubbles=ordered_bubbles,
        processing_time_ms=duration_ms
    )


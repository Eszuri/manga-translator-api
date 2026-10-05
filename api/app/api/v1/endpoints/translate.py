import asyncio
import base64
import io
import json
import time
from typing import Literal
from fastapi import APIRouter, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse

from app.api.v1.image_uploads import read_validated_image
from app.schemas import (
    DetectBubblesResponse,
    InpaintPageResponse,
    TranslateDialoguesRequest,
    TranslateDialoguesResponse,
    TranslatedDialogueItem
)
from app.core.gpu import require_gpu_device
from app.core.config import settings
from app.core.image_worker import run_image_task, stream_with_heartbeats
from app.services.detector import sort_manga_reading_order
from app.services.ocr_service import get_ocr_service
from app.services.translation_service import TranslationError, get_translation_service
from app.services.inpainting_service import MangaInpaintingService
from app.services.typesetting_service import MangaTypesettingService

router = APIRouter()

_comic_detector = None
_hybrid_detector = None


def get_detector_instance(detector_type: str, device: str):
    global _comic_detector, _hybrid_detector
    require_gpu_device(device)
    if detector_type == "comic_text_detector":
        if _comic_detector is None:
            from app.services.comic_text_detector import ComicTextDetector
            _comic_detector = ComicTextDetector(device="gpu")
        return _comic_detector
    elif detector_type == "hybrid":
        if _hybrid_detector is None:
            from app.services.hybrid_detector import HybridBubbleDetector
            from app.services.comic_text_detector import ComicTextDetector
            comic_det = ComicTextDetector(device="gpu")
            _hybrid_detector = HybridBubbleDetector(comic_detector=comic_det)
        return _hybrid_detector
    raise ValueError("GPU-only backend does not support the CPU contour detector.")


def detect_page(image, detector_type, device, reading_direction, include_seg=False):
    detector = get_detector_instance(detector_type, device)
    detected = detector.detect(image)
    bubbles = sort_manga_reading_order(detected, reading_direction=reading_direction)
    seg_mask = None
    if include_seg:
        comic_det = getattr(detector, "comic_detector", detector)
        if hasattr(comic_det, "detect_raw") and hasattr(comic_det, "get_unletterboxed_seg"):
            _, seg, _, _, (dw, dh) = comic_det.detect_raw(image)
            seg_mask = comic_det.get_unletterboxed_seg(seg, image.width, image.height, dw, dh)
    return bubbles, seg_mask


def recognize_bubbles(image, bubbles, device):
    return get_ocr_service(device=device).recognize_all_bubbles(image, bubbles)


def require_bubble_text(bubbles, translated=False):
    field = "translation" if translated else "text"
    missing = [str(b.id) for b in bubbles if not (getattr(b, field) or "").strip()]
    if missing:
        stage = "Translation" if translated else "OCR"
        raise TranslationError(
            f"{stage} returned empty text for bubble(s): {', '.join(missing)}. "
            "The original image was not modified."
        )


def encode_image(image):
    out_buf = io.BytesIO()
    image.convert("RGB").save(out_buf, format="JPEG", quality=95)
    return out_buf.getvalue()


@router.post("/page", response_model=DetectBubblesResponse)
async def translate_manga_page(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    target_lang: str = Form(
        "id",
        description="Target translation language code ('id' for Indonesian, 'en' for English)"
    ),
    detector_type: Literal["hybrid", "comic_text_detector"] = Form(
        "hybrid",
        description="Text/bubble detector engine ('hybrid' recommended for optimal accuracy)"
    ),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
    ),
    device: Literal["gpu"] = Form(
        "gpu",
        description="GPU-only backend; the only accepted value is 'gpu'"
    )
):
    """
    Complete End-to-End Manga Translation Pipeline:
    1. Speech Bubble & Text Detection (Hybrid AI + Balloon Segmentation).
    2. Reading Order Sorting (Right-to-Left / Left-to-Right).
    3. Japanese Text Recognition via Manga-OCR (ViT Transformer).
    4. Contextual Dialogue Translation via OpenAI-compatible LLM (default: Indonesian).
    """
    image = await read_validated_image(file)

    start_time = time.perf_counter()

    ordered_bubbles, _ = await run_image_task(
        detect_page, image, detector_type, device, reading_direction
    )
    ordered_bubbles = await run_image_task(recognize_bubbles, image, ordered_bubbles, device)

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
    translator: Literal["llm", "google"] = Form(
        settings.DEFAULT_TRANSLATOR,
        description="Translation engine: 'llm' (OpenAI/Ollama) or 'google' (Google Translate)"
    ),
    detector_type: Literal["hybrid", "comic_text_detector"] = Form(
        "hybrid",
        description="Text/bubble detector engine ('hybrid' recommended)"
    ),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
    ),
    device: Literal["gpu"] = Form(
        "gpu",
        description="GPU-only backend; the only accepted value is 'gpu'"
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
    image = await read_validated_image(file)

    start_time = time.perf_counter()

    ordered_bubbles, seg_mask = await run_image_task(
        detect_page, image, detector_type, device, reading_direction, include_seg=True
    )

    if typeset and ordered_bubbles:
        ordered_bubbles = await run_image_task(recognize_bubbles, image, ordered_bubbles, device)
        require_bubble_text(ordered_bubbles)

        trans_service = get_translation_service()
        ordered_bubbles = await trans_service.translate_bubbles_async(
            ordered_bubbles, target_lang=target_lang, translator=translator
        )
        require_bubble_text(ordered_bubbles, translated=True)

    inpaint_service = MangaInpaintingService()
    inpainted_img = await run_image_task(
        inpaint_service.inpaint, image, seg_mask=seg_mask, bubbles=ordered_bubbles
    )

    if typeset and ordered_bubbles:
        typeset_service = MangaTypesettingService(all_caps=all_caps)
        final_img = await run_image_task(
            typeset_service.typeset, inpainted_img, ordered_bubbles, font_scale=font_scale
        )
    else:
        final_img = inpainted_img

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    img_bytes = await run_image_task(encode_image, final_img)

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


@router.post("/inpaint-stream")
async def inpaint_stream_manga_page(
    file: UploadFile = File(..., description="Manga page image file (JPEG, PNG, WebP)"),
    target_lang: str = Form(
        "id",
        description="Target translation language code ('id' for Indonesian, 'en' for English)"
    ),
    translator: Literal["llm", "google"] = Form(
        settings.DEFAULT_TRANSLATOR,
        description="Translation engine: 'llm' (OpenAI/Ollama) or 'google' (Google Translate)"
    ),
    detector_type: Literal["hybrid", "comic_text_detector"] = Form(
        "hybrid",
        description="Text/bubble detector engine ('hybrid' recommended)"
    ),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
    ),
    device: Literal["gpu"] = Form(
        "gpu",
        description="GPU-only backend; the only accepted value is 'gpu'"
    ),
    typeset: bool = Form(
        True,
        description="Whether to typeset translated text into bubbles"
    ),
    font_scale: float = Form(
        1.0,
        description="Font size multiplier (default: 1.0)"
    ),
    all_caps: bool = Form(
        True,
        description="Render dialogue text in comic uppercase (default: True)"
    )
):
    """
    Streaming inpainting & typesetting endpoint:
    Yields real-time NDJSON events for each stage (detect, ocr, translate, inpaint, render, done).
    """
    image = await read_validated_image(file)

    async def stream_generator():
        start_time = time.perf_counter()
        try:
            yield json.dumps({"stage": "detect", "message": "Detecting text bubbles..."}) + "\n"
            await asyncio.sleep(0.01)

            ordered_bubbles, seg_mask = await run_image_task(
                detect_page, image, detector_type, device, reading_direction, include_seg=True
            )

            if typeset and ordered_bubbles:
                yield json.dumps({
                    "stage": "ocr",
                    "message": f"Reading OCR text ({len(ordered_bubbles)} bubbles)...",
                    "total_bubbles": len(ordered_bubbles)
                }) + "\n"
                await asyncio.sleep(0.01)

                ordered_bubbles = await run_image_task(recognize_bubbles, image, ordered_bubbles, device)
                require_bubble_text(ordered_bubbles)

                yield json.dumps({
                    "stage": "translate",
                    "message": f"Translating {len(ordered_bubbles)} dialogues...",
                    "total_bubbles": len(ordered_bubbles)
                }) + "\n"
                await asyncio.sleep(0.01)

                trans_service = get_translation_service()
                ordered_bubbles = await trans_service.translate_bubbles_async(
                    ordered_bubbles, target_lang=target_lang, translator=translator
                )
                require_bubble_text(ordered_bubbles, translated=True)
            else:
                yield json.dumps({
                    "stage": "ocr",
                    "message": "No text bubbles found...",
                    "total_bubbles": 0
                }) + "\n"
                await asyncio.sleep(0.01)

            yield json.dumps({"stage": "inpaint", "message": "Removing original text (inpainting)..."}) + "\n"
            await asyncio.sleep(0.01)

            inpaint_service = MangaInpaintingService()
            inpainted_img = await run_image_task(
                inpaint_service.inpaint, image, seg_mask=seg_mask, bubbles=ordered_bubbles
            )

            yield json.dumps({"stage": "render", "message": "Rendering and typesetting text..."}) + "\n"
            await asyncio.sleep(0.01)

            if typeset and ordered_bubbles:
                typeset_service = MangaTypesettingService(all_caps=all_caps)
                final_img = await run_image_task(
                    typeset_service.typeset, inpainted_img, ordered_bubbles, font_scale=font_scale
                )
            else:
                final_img = inpainted_img

            img_bytes = await run_image_task(encode_image, final_img)
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            yield json.dumps({
                "stage": "done",
                "message": "Complete",
                "image_base64": f"data:image/jpeg;base64,{img_b64}",
                "total_detected": len(ordered_bubbles),
                "duration_ms": duration_ms
            }) + "\n"

        except Exception as e:
            yield json.dumps({"stage": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        stream_with_heartbeats(stream_generator()), media_type="application/x-ndjson"
    )

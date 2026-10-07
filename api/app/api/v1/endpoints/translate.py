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
from app.core.config import settings
from app.core.image_worker import run_image_task, stream_with_heartbeats
from app.services.detector import sort_manga_reading_order
from app.services.ocr_service import get_ocr_service
from app.services.translation_service import get_translation_service
from app.services.translation_filters import is_graphic_text, usable_translation
from app.services.inpainting_service import MangaInpaintingService
from app.services.typesetting_service import MangaTypesettingService

router = APIRouter()

_hybrid_detector = None


def get_hybrid_detector():
    global _hybrid_detector
    if _hybrid_detector is None:
        from app.services.hybrid_detector import HybridBubbleDetector
        _hybrid_detector = HybridBubbleDetector()
    return _hybrid_detector


def detect_page(image, reading_direction, include_seg=False):
    detector = get_hybrid_detector()
    detected = detector.detect(image)
    bubbles = sort_manga_reading_order(detected, reading_direction=reading_direction)
    seg_mask = None
    if include_seg:
        comic_det = getattr(detector, "comic_detector", detector)
        if hasattr(comic_det, "detect_raw") and hasattr(comic_det, "get_unletterboxed_seg"):
            _, seg, _, _, (dw, dh) = comic_det.detect_raw(image)
            seg_mask = comic_det.get_unletterboxed_seg(seg, image.width, image.height, dw, dh)
    return bubbles, seg_mask


def recognize_bubbles(image, bubbles):
    return get_ocr_service().recognize_all_bubbles(image, bubbles)


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
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
    )
):
    image = await read_validated_image(file)

    start_time = time.perf_counter()

    ordered_bubbles, _ = await run_image_task(
        detect_page, image, reading_direction
    )
    ordered_bubbles = await run_image_task(recognize_bubbles, image, ordered_bubbles)

    candidates = [b for b in ordered_bubbles if (b.text or "").strip() and not is_graphic_text(b.text)]
    if candidates:
        trans_service = get_translation_service()
        await trans_service.translate_bubbles_async(candidates, target_lang=target_lang)

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
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
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
    image = await read_validated_image(file)

    start_time = time.perf_counter()

    ordered_bubbles, seg_mask = await run_image_task(
        detect_page, image, reading_direction, include_seg=True
    )

    active_bubbles = []
    if typeset and ordered_bubbles:
        ordered_bubbles = await run_image_task(recognize_bubbles, image, ordered_bubbles)
        candidates = [b for b in ordered_bubbles if (b.text or "").strip() and not is_graphic_text(b.text)]

        if candidates:
            trans_service = get_translation_service()
            await trans_service.translate_bubbles_async(
                candidates, target_lang=target_lang, translator=translator
            )
            active_bubbles = [b for b in candidates if usable_translation(b.translation or "")]

    inpaint_service = MangaInpaintingService()
    inpainted_img = await run_image_task(
        inpaint_service.inpaint, image, seg_mask=seg_mask, bubbles=active_bubbles or ordered_bubbles
    )

    if typeset and active_bubbles:
        typeset_service = MangaTypesettingService(all_caps=all_caps)
        final_img = await run_image_task(
            typeset_service.typeset, inpainted_img, active_bubbles, font_scale=font_scale
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
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Reading direction ('rtl' for Japanese Manga, 'ltr' for Manhwa)"
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
    image = await read_validated_image(file)

    async def stream_generator():
        start_time = time.perf_counter()
        try:
            yield json.dumps({"stage": "detect", "message": "Detecting text bubbles..."}) + "\n"
            await asyncio.sleep(0.01)

            ordered_bubbles, seg_mask = await run_image_task(
                detect_page, image, reading_direction, include_seg=True
            )

            active_bubbles = []
            if typeset and ordered_bubbles:
                yield json.dumps({
                    "stage": "ocr",
                    "message": f"Reading OCR text ({len(ordered_bubbles)} bubbles)...",
                    "total_bubbles": len(ordered_bubbles)
                }) + "\n"
                await asyncio.sleep(0.01)

                ordered_bubbles = await run_image_task(recognize_bubbles, image, ordered_bubbles)
                candidates = [b for b in ordered_bubbles if (b.text or "").strip() and not is_graphic_text(b.text)]

                if candidates:
                    yield json.dumps({
                        "stage": "translate",
                        "message": f"Translating {len(candidates)} dialogues...",
                        "total_bubbles": len(candidates)
                    }) + "\n"
                    await asyncio.sleep(0.01)

                    trans_service = get_translation_service()
                    await trans_service.translate_bubbles_async(
                        candidates, target_lang=target_lang, translator=translator
                    )
                    active_bubbles = [b for b in candidates if usable_translation(b.translation or "")]
                else:
                    yield json.dumps({
                        "stage": "translate",
                        "message": "No translatable text found...",
                        "total_bubbles": 0
                    }) + "\n"
                    await asyncio.sleep(0.01)
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
                inpaint_service.inpaint, image, seg_mask=seg_mask, bubbles=active_bubbles or ordered_bubbles
            )

            yield json.dumps({"stage": "render", "message": "Rendering and typesetting text..."}) + "\n"
            await asyncio.sleep(0.01)

            if typeset and active_bubbles:
                typeset_service = MangaTypesettingService(all_caps=all_caps)
                final_img = await run_image_task(
                    typeset_service.typeset, inpainted_img, active_bubbles, font_scale=font_scale
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

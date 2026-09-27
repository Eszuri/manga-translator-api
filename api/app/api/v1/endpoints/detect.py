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
    file: UploadFile = File(..., description="File gambar halaman manga (JPEG, PNG, WebP)"),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl", 
        description="Arah baca: 'rtl' (Manga Jepang) atau 'ltr' (Manhwa Korea / Webtoon)"
    ),
    detector_type: Literal["contour", "comic_text_detector", "hybrid"] = Form(
        "contour",
        description="Tipe detektor: 'contour' (cepat OpenCV), 'comic_text_detector' (model AI), atau 'hybrid' (AI + Balon OpenCV)"
    )
):
    """
    Mendeteksi balon percakapan (speech bubbles) pada gambar halaman manga.
    Mengembalikan daftar koordinat bounding box yang sudah diurutkan berdasarkan kaidah baca manga.
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp", "application/octet-stream"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Format file '{file.content_type}' tidak didukung. Gunakan JPEG, PNG, atau WebP."
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File gambar yang diunggah kosong."
            )
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gagal memproses file gambar: {str(e)}"
        )

    start_time = time.perf_counter()

    # 1. Jalankan deteksi balon kata / teks
    if detector_type == "comic_text_detector":
        active_detector = get_comic_text_detector()
    elif detector_type == "hybrid":
        active_detector = get_hybrid_detector()
    else:
        active_detector = contour_detector

    detected = active_detector.detect(image)

    # 2. Urutkan sesuai kaidah baca komik (RTL / LTR)
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

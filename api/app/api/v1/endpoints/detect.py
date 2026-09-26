import io
import time
from typing import Literal
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from PIL import Image

from app.schemas import DetectBubblesResponse
from app.services.detector import ContourBubbleDetector, sort_manga_reading_order

router = APIRouter()

detector = ContourBubbleDetector()


@router.post("/bubbles", response_model=DetectBubblesResponse)
async def detect_bubbles(
    file: UploadFile = File(..., description="File gambar halaman manga (JPEG, PNG, WebP)"),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl", 
        description="Arah baca: 'rtl' (Manga Jepang) atau 'ltr' (Manhwa Korea / Webtoon)"
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

    # 1. Jalankan deteksi balon kata
    detected = detector.detect(image)

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

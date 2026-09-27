import io
import time
from typing import Literal, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field
from PIL import Image

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
    text: str = Field(..., description="Teks hasil ekstraksi Manga OCR")
    processing_time_ms: float


@router.post("/recognize", response_model=DetectBubblesResponse)
async def recognize_page_text(
    file: UploadFile = File(..., description="Gambar halaman manga (JPEG, PNG, WebP)"),
    detector_type: Literal["hybrid", "comic_text_detector", "contour"] = Form(
        "hybrid",
        description="Detektor teks/balon ('hybrid' direkomendasikan untuk akurasi optimal)"
    ),
    reading_direction: Literal["rtl", "ltr"] = Form(
        "rtl",
        description="Arah baca ('rtl' untuk Manga Jepang, 'ltr' untuk Manhwa)"
    ),
    device: Literal["auto", "gpu", "cpu"] = Form(
        "auto",
        description="Perangkat komputasi: 'gpu' (DirectML), 'cpu', atau 'auto'"
    )
):
    """
    Ekstraksi teks lengkap pada satu halaman manga:
    1. Mendeteksi letak balon kata dan area teks secara otomatis.
    2. Mengurutkan balon berdasarkan kaidah baca manga (RTL).
    3. Memotong setiap area teks dan mengekstrak teks Jepang menggunakan Manga-OCR (ViT).
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

    # 1. Deteksi balon kata
    detector = get_detector_instance(detector_type, device)
    detected = detector.detect(image)

    # 2. Urutkan berdasarkan arah baca
    ordered_bubbles = sort_manga_reading_order(detected, reading_direction=reading_direction)

    # 3. Jalankan Manga-OCR pada setiap teks yang terdeteksi
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
    file: UploadFile = File(..., description="Gambar potongan teks (crop) dari seleksi user"),
    device: Literal["auto", "gpu", "cpu"] = Form(
        "auto",
        description="Perangkat komputasi: 'gpu' (DirectML), 'cpu', atau 'auto'"
    )
):
    """
    Ekstraksi teks langsung dari potongan gambar (fitur seleksi manual / drag box dari extension).
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
                detail="File gambar crop yang diunggah kosong."
            )
        crop_img = Image.open(io.BytesIO(contents))
        crop_img.load()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gagal memproses gambar crop: {str(e)}"
        )

    start_time = time.perf_counter()
    ocr_service = get_ocr_service(device=device)
    text = ocr_service.recognize_crop(crop_img)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return CropOCRResponse(
        success=True,
        text=text,
        processing_time_ms=duration_ms
    )

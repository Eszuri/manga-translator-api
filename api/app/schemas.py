from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(..., description="Koordinat X sudut kiri-atas (piksel)", ge=0)
    y: int = Field(..., description="Koordinat Y sudut kiri-atas (piksel)", ge=0)
    width: int = Field(..., description="Lebar kotak (piksel)", gt=0)
    height: int = Field(..., description="Tinggi kotak (piksel)", gt=0)

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height


class DetectedBubble(BaseModel):
    id: int = Field(..., description="Nomor urut balon kata sesuai urutan baca")
    bounding_box: BoundingBox
    text_box: Optional[BoundingBox] = Field(
        default=None, 
        description="Koordinat kotak teks di dalam balon (terpisah dari batas balon)"
    )
    text: Optional[str] = Field(
        default=None,
        description="Teks dialog hasil ekstraksi Manga OCR"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Tingkat kepercayaan deteksi")
    direction: Literal["vertical", "horizontal"] = Field(
        default="vertical", 
        description="Estimasi orientasi teks ('vertical' untuk Manga Jepang, 'horizontal' untuk Manhwa)"
    )
    aspect_ratio: float = Field(..., description="Rasio tinggi / lebar kotak (height / width)")
    detector_type: Optional[str] = Field(
        default="contour", 
        description="Tipe detektor yang mendeteksi ('contour', 'comic_text_detector', 'hybrid')"
    )


class DetectBubblesResponse(BaseModel):
    success: bool = True
    image_width: int
    image_height: int
    total_detected: int
    reading_direction: Literal["rtl", "ltr"] = Field(
        default="rtl", 
        description="'rtl' (Kanan-ke-Kiri untuk Manga) atau 'ltr' (Kiri-ke-Kanan untuk Manhwa)"
    )
    bubbles: List[DetectedBubble] = Field(default_factory=list)
    processing_time_ms: float

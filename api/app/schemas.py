from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left X coordinate (pixels)", ge=0)
    y: int = Field(..., description="Top-left Y coordinate (pixels)", ge=0)
    width: int = Field(..., description="Box width (pixels)", gt=0)
    height: int = Field(..., description="Box height (pixels)", gt=0)

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
    id: int = Field(..., description="Speech bubble index in reading order")
    bounding_box: BoundingBox
    text_box: Optional[BoundingBox] = Field(
        default=None, 
        description="Inner text envelope coordinates inside the speech bubble"
    )
    text: Optional[str] = Field(
        default=None,
        description="Extracted Japanese dialogue text from Manga OCR"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score")
    direction: Literal["vertical", "horizontal"] = Field(
        default="vertical", 
        description="Estimated text reading orientation ('vertical' for Manga, 'horizontal' for Manhwa)"
    )
    aspect_ratio: float = Field(..., description="Height-to-width ratio (height / width)")
    detector_type: Optional[str] = Field(
        default="hybrid", 
        description="Detector engine used ('hybrid', 'comic_text_detector', 'contour')"
    )


class DetectBubblesResponse(BaseModel):
    success: bool = True
    image_width: int
    image_height: int
    total_detected: int
    reading_direction: Literal["rtl", "ltr"] = Field(
        default="rtl", 
        description="'rtl' (Right-to-Left for Japanese Manga) or 'ltr' (Left-to-Right for Manhwa)"
    )
    bubbles: List[DetectedBubble] = Field(default_factory=list)
    processing_time_ms: float

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
    translation: Optional[str] = Field(
        default=None,
        description="Translated dialogue text into target language (default: Indonesian)"
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
    target_lang: Optional[str] = Field(
        default=None,
        description="Target language code for dialogue translation (e.g. 'id', 'en')"
    )
    bubbles: List[DetectedBubble] = Field(default_factory=list)
    processing_time_ms: float


class DialogueItem(BaseModel):
    id: int = Field(..., description="Speech bubble index ID")
    text: str = Field(..., description="Original Japanese text")


class TranslateDialoguesRequest(BaseModel):
    dialogues: List[DialogueItem] = Field(..., description="List of dialogues to translate in reading order")
    target_lang: str = Field(default="id", description="Target language code ('id' for Indonesian, 'en' for English)")
    context: Optional[str] = Field(default=None, description="Optional scene context or manga genre info")


class TranslatedDialogueItem(BaseModel):
    id: int
    original_text: str
    translated_text: str


class TranslateDialoguesResponse(BaseModel):
    success: bool = True
    target_lang: str
    dialogues: List[TranslatedDialogueItem] = Field(default_factory=list)
    processing_time_ms: float
    model: str


class InpaintPageResponse(BaseModel):
    success: bool = True
    image_width: int
    image_height: int
    total_detected: int
    target_lang: Optional[str] = None
    image_base64: str = Field(..., description="Base64-encoded clean or typeset manga image")
    bubbles: List[DetectedBubble] = Field(default_factory=list)
    processing_time_ms: float



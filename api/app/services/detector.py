from abc import ABC, abstractmethod
from typing import List
from PIL import Image, ImageDraw

from app.schemas import DetectedBubble


def sort_manga_reading_order(
    bubbles: List[DetectedBubble],
    reading_direction: str = "rtl",
    row_tolerance_ratio: float = 0.35
) -> List[DetectedBubble]:
    """
    Sorts detected speech bubbles in authentic comic reading order:
    - RTL (Manga): Top-to-Bottom by panel row band, then Right-to-Left within each row.
    - LTR (Manhwa / Western): Top-to-Bottom by row, then Left-to-Right.
    """
    if not bubbles:
        return []

    rows = []
    for bubble in sorted(bubbles, key=lambda b: b.bounding_box.y):
        box = bubble.bounding_box
        if rows:
            row = rows[-1]
            overlap = min(row['bottom'], box.bottom) - max(row['top'], box.y)
            if overlap >= max(1, min(box.height, row['bottom'] - row['top']) * row_tolerance_ratio):
                row['bubbles'].append(bubble)
                row['bottom'] = max(row['bottom'], box.bottom)
                continue
        rows.append(dict(top=box.y, bottom=box.bottom, bubbles=[bubble]))
    sign = 1 if reading_direction == 'ltr' else -1
    sorted_list = [b for row in rows for b in sorted(
        row['bubbles'], key=lambda b: (sign * b.bounding_box.center_x, b.bounding_box.y))]

    for idx, bubble in enumerate(sorted_list, start=1):
        bubble.id = idx

    return sorted_list


class BaseBubbleDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> List[DetectedBubble]:
        """Detect speech bubbles from a PIL Image."""


def annotate_and_save_bubbles(
    image: Image.Image,
    bubbles: List[DetectedBubble],
    output_path: str,
    draw_text_boxes: bool = True,
    draw_layout: bool = False
) -> str:
    """
    Renders bounding box annotations onto a copy of the manga image
    and saves exactly one output file at the specified output_path.
    Draws outer bubble in red and inner text box in green (if present and distinct).
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for b in bubbles:
        box = b.bounding_box
        draw.rectangle([box.x, box.y, box.right, box.bottom], outline=(255, 0, 0), width=4)
        draw.rectangle([box.x, max(0, box.y - 28), box.x + 44, box.y], fill=(255, 0, 0))
        draw.text((box.x + 8, max(0, box.y - 24)), f"#{b.id}", fill=(255, 255, 255))

        if draw_text_boxes and b.text_box is not None:
            t = b.text_box
            if t.x != box.x or t.y != box.y or t.width != box.width or t.height != box.height:
                draw.rectangle([t.x, t.y, t.right, t.bottom], outline=(0, 200, 0), width=2)

        if draw_layout:
            if b.bubble_polygon:
                points = [tuple(p) for p in b.bubble_polygon]
                draw.line(points + points[:1], fill=(255, 140, 0), width=2)
            if b.layout_box:
                t = b.layout_box
                draw.rectangle([t.x, t.y, t.right, t.bottom], outline=(0, 100, 255), width=2)

    out.save(output_path)
    return output_path

import os
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from app.schemas import DetectedBubble


DEFAULT_FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "fonts",
    "comic_bold.ttf"
)


class MangaTypesettingService:
    """
    Typesetting service for manga dialogue translation.
    Features:
    - Dynamic font scaling with auto-fit binary search.
    - Balanced, elliptical word-wrapping tailored for comic speech balloons.
    - Centered multi-line alignment.
    - Adaptive text stroke/outline for maximum legibility on any background.
    """

    def __init__(
        self,
        font_path: Optional[str] = None,
        padding_ratio: float = 0.22,
        line_spacing_ratio: float = 0.20,
        all_caps: bool = True
    ):
        self.font_path = font_path or DEFAULT_FONT_PATH
        self.padding_ratio = padding_ratio
        self.line_spacing_ratio = line_spacing_ratio
        self.all_caps = all_caps
        self._font_cache = {}

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Retrieves or creates a cached TrueType font instance at the specified size."""
        if size in self._font_cache:
            return self._font_cache[size]

        font = None
        # 1. Try specified comic font path
        if self.font_path and os.path.exists(self.font_path):
            try:
                font = ImageFont.truetype(self.font_path, size=size)
            except Exception:
                font = None

        # 2. Try Windows system Comic Sans / Arial fallbacks
        if font is None:
            fallbacks = [
                "C:\\Windows\\Fonts\\comicbd.ttf",
                "C:\\Windows\\Fonts\\comic.ttf",
                "C:\\Windows\\Fonts\\arialbd.ttf",
                "C:\\Windows\\Fonts\\arial.ttf"
            ]
            for fb in fallbacks:
                if os.path.exists(fb):
                    try:
                        font = ImageFont.truetype(fb, size=size)
                        break
                    except Exception:
                        continue

        # 3. Final default fallback
        if font is None:
            font = ImageFont.load_default()

        self._font_cache[size] = font
        return font

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.ImageFont,
        max_width: int,
        draw: ImageDraw.ImageDraw
    ) -> List[str]:
        """
        Wraps text into lines using pixel-accurate bounding box measurements.
        Balances line lengths to avoid single-word dangling orphans.
        """
        words = text.strip().split()
        if not words:
            return []

        lines: List[str] = []
        current_line: List[str] = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_w = bbox[2] - bbox[0]

            if line_w <= max_width or not current_line:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        # Balance last line if it contains only 1 small word and previous line has >= 3 words
        if len(lines) >= 2 and len(lines[-1].split()) == 1:
            prev_words = lines[-2].split()
            if len(prev_words) >= 3:
                moved_word = prev_words.pop()
                lines[-2] = " ".join(prev_words)
                lines[-1] = f"{moved_word} {lines[-1]}"

        return lines

    def _measure_text_block(
        self,
        lines: List[str],
        font: ImageFont.ImageFont,
        draw: ImageDraw.ImageDraw
    ) -> Tuple[int, int, List[int]]:
        """
        Calculates total width, total height, and individual line heights for a wrapped block.
        """
        if not lines:
            return 0, 0, []

        line_heights = []
        max_w = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            max_w = max(max_w, w)
            line_heights.append(max(h, font.size))

        spacing = int(font.size * self.line_spacing_ratio)
        total_h = sum(line_heights) + max(0, len(lines) - 1) * spacing
        return max_w, total_h, line_heights

    def find_optimal_font_and_lines(
        self,
        text: str,
        target_w: int,
        target_h: int,
        draw: ImageDraw.ImageDraw,
        page_scale: float = 1.0,
        font_scale: float = 1.0
    ) -> Tuple[ImageFont.ImageFont, List[str], List[int], int]:
        """
        Uses binary search to find the largest font size where the wrapped text
        comfortably fits inside target dimensions.
        """
        min_size = max(8, int(10 * page_scale * font_scale))
        max_size = max(min_size + 4, int(46 * page_scale * font_scale))

        best_font = self._get_font(min_size)
        best_lines = self._wrap_text(text, best_font, target_w, draw)
        best_w, best_h, best_line_heights = self._measure_text_block(best_lines, best_font, draw)
        best_spacing = int(best_font.size * self.line_spacing_ratio)

        low = min_size
        high = max_size

        while low <= high:
            mid = (low + high) // 2
            test_font = self._get_font(mid)
            lines = self._wrap_text(text, test_font, target_w, draw)
            w, h, line_heights = self._measure_text_block(lines, test_font, draw)

            if w <= target_w and h <= target_h:
                best_font = test_font
                best_lines = lines
                best_line_heights = line_heights
                best_spacing = int(test_font.size * self.line_spacing_ratio)
                low = mid + 1
            else:
                high = mid - 1

        return best_font, best_lines, best_line_heights, best_spacing

    def typeset(
        self,
        image: Image.Image,
        bubbles: List[DetectedBubble],
        font_scale: float = 1.0,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        stroke_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Renders translated dialogue text inside speech bubbles with comic typography.
        """
        if not bubbles:
            return image

        output_img = image.copy()
        draw = ImageDraw.Draw(output_img)

        page_scale = max(image.width, image.height) / 1200.0

        for bubble in bubbles:
            text = bubble.translation or bubble.text
            if not text or not text.strip():
                continue

            if self.all_caps:
                text = text.upper()

            # Target speech balloon boundary
            bbox = bubble.bounding_box
            bw = bbox.width
            bh = bbox.height

            # Adaptive padding: tall/spiky bubbles need extra horizontal safety margin
            aspect = bh / max(1.0, float(bw))
            pad_x_ratio = 0.28 if aspect > 1.3 else self.padding_ratio
            pad_y_ratio = self.padding_ratio

            margin_x = int(bw * (pad_x_ratio / 2.0))
            margin_y = int(bh * (pad_y_ratio / 2.0))
            target_w = max(20, bw - 2 * margin_x)
            target_h = max(20, bh - 2 * margin_y)

            # Find optimal font and wrapped lines
            font, lines, line_heights, spacing = self.find_optimal_font_and_lines(
                text=text,
                target_w=target_w,
                target_h=target_h,
                draw=draw,
                page_scale=page_scale,
                font_scale=font_scale
            )

            if not lines:
                continue

            # Compute vertical starting point (centered in bubble)
            total_h = sum(line_heights) + max(0, len(lines) - 1) * spacing
            cur_y = bbox.center_y - total_h / 2.0

            # Dynamic stroke thickness for crispness
            stroke_width = max(1, int(font.size * 0.07)) if stroke_color else 0

            for i, line in enumerate(lines):
                line_bbox = draw.textbbox((0, 0), line, font=font)
                lw = line_bbox[2] - line_bbox[0]
                line_x = bbox.center_x - lw / 2.0

                draw.text(
                    (line_x, cur_y),
                    line,
                    font=font,
                    fill=text_color,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color if stroke_width > 0 else None
                )
                cur_y += line_heights[i] + spacing

        return output_img

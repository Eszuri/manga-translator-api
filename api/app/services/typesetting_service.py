import os
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageChops

from app.schemas import DetectedBubble


DEFAULT_FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "fonts",
    "comic_bold.ttf"
)


class TypesettingError(RuntimeError):
    """A dialogue cannot be rendered safely; never return an erased bubble."""


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
        if self.font_path and os.path.exists(self.font_path):
            try:
                font = ImageFont.truetype(self.font_path, size=size)
            except Exception:
                font = None

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

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[size] = font
        return font

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.ImageFont,
        max_width: int,
        draw: ImageDraw.ImageDraw,
        break_long_words: bool = True
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

        def split_long_word(word: str) -> List[str]:
            """Split an unspaced token into pixel-width-safe fragments."""
            fragments: List[str] = []
            fragment = ""
            for character in word:
                candidate = fragment + character
                candidate_bbox = draw.textbbox((0, 0), candidate, font=font)
                candidate_width = candidate_bbox[2] - candidate_bbox[0]
                if fragment and candidate_width > max_width:
                    fragments.append(fragment)
                    fragment = character
                else:
                    fragment = candidate
            if fragment:
                fragments.append(fragment)
            return fragments

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_w = bbox[2] - bbox[0]

            if line_w <= max_width:
                current_line.append(word)
            elif not current_line:
                fragments = split_long_word(word) if break_long_words else [word]
                lines.extend(fragments[:-1])
                current_line = fragments[-1:]
            else:
                lines.append(" ".join(current_line))
                word_bbox = draw.textbbox((0, 0), word, font=font)
                word_width = word_bbox[2] - word_bbox[0]
                if word_width > max_width:
                    fragments = split_long_word(word) if break_long_words else [word]
                    lines.extend(fragments[:-1])
                    current_line = fragments[-1:]
                else:
                    current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        if len(lines) >= 2 and len(lines[-1].split()) == 1:
            prev_words = lines[-2].split()
            if len(prev_words) >= 3:
                moved_word = prev_words.pop()
                candidate_last_line = f"{moved_word} {lines[-1]}"
                candidate_bbox = draw.textbbox((0, 0), candidate_last_line, font=font)
                candidate_width = candidate_bbox[2] - candidate_bbox[0]
                if candidate_width <= max_width:
                    lines[-2] = " ".join(prev_words)
                    lines[-1] = candidate_last_line

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
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=max(1, int(font.size * 0.07)))
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
        page_cap = 28 * page_scale * font_scale
        width_cap = max(12 * page_scale, target_w * 0.25) * font_scale
        max_size = max(1, min(target_h, int(page_cap), int(width_cap)))
        best_font, best_lines, best_line_heights, best_spacing = self._get_font(1), [], [], 0
        low = 1
        high = max_size

        while low <= high:
            mid = (low + high) // 2
            test_font = self._get_font(mid)
            stroke_width = max(1, int(test_font.size * 0.07))
            lines = self._wrap_text(text, test_font, max(1, target_w - 2 * stroke_width), draw,
                                    break_long_words=False)
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

        page_scale = min(image.width / 900.0, image.height / 1200.0)

        for bubble in bubbles:
            text = bubble.translation or bubble.text
            if not text or not text.strip():
                raise TypesettingError(f'Bubble {bubble.id}: no text available for rendering.')

            if self.all_caps:
                text = text.upper()

            bbox = bubble.layout_box or bubble.text_box or bubble.bounding_box
            bw = bbox.width
            bh = bbox.height

            aspect = bh / max(1.0, float(bw))
            pad_x_ratio = 0.06 if bubble.layout_box else (0.28 if aspect > 1.3 else self.padding_ratio)
            pad_y_ratio = 0.06 if bubble.layout_box else self.padding_ratio

            margin_x = int(bw * (pad_x_ratio / 2.0))
            margin_y = int(bh * (pad_y_ratio / 2.0))
            content_left = max(0, bbox.x + margin_x)
            content_top = max(0, bbox.y + margin_y)
            content_right = min(output_img.width, bbox.right - margin_x)
            content_bottom = min(output_img.height, bbox.bottom - margin_y)

            if content_right <= content_left or content_bottom <= content_top:
                raise TypesettingError(f'Bubble {bubble.id}: render region is outside the image or empty.')

            target_w = content_right - content_left
            target_h = content_bottom - content_top

            region_font_scale = font_scale * (0.65 if not bubble.bubble_polygon else 1.0)
            font, lines, line_heights, spacing = self.find_optimal_font_and_lines(
                text=text,
                target_w=target_w,
                target_h=target_h,
                draw=draw,
                page_scale=page_scale,
                font_scale=region_font_scale
            )

            if not lines:
                raise TypesettingError(f'Bubble {bubble.id}: text does not fit the render region without truncation.')

            text_layer = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_layer)

            total_h = sum(line_heights) + max(0, len(lines) - 1) * spacing
            cur_y = (target_h - total_h) / 2.0

            stroke_width = max(1, int(font.size * 0.07)) if stroke_color else 0

            for i, line in enumerate(lines):
                line_bbox = text_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
                lw = line_bbox[2] - line_bbox[0]
                line_x = (target_w - lw) / 2.0 - line_bbox[0]

                text_draw.text(
                    (line_x, cur_y - line_bbox[1]),
                    line,
                    font=font,
                    fill=text_color,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color if stroke_width > 0 else None
                )
                cur_y += line_heights[i] + spacing

            if bubble.bubble_polygon:
                shape = Image.new('L', text_layer.size, 0)
                ImageDraw.Draw(shape).polygon(
                    [(x - content_left, y - content_top) for x, y in bubble.bubble_polygon], fill=255)
                text_layer.putalpha(ImageChops.multiply(text_layer.getchannel('A'), shape))
            if text_layer.getchannel('A').getbbox() is None:
                raise TypesettingError(f'Bubble {bubble.id}: the bubble mask hides all rendered text.')
            output_img.paste(text_layer, (content_left, content_top), text_layer)

        return output_img

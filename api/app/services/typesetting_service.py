import os
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.schemas import DetectedBubble


DEFAULT_FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "fonts",
    "comic_bold.ttf"
)


class TypesettingError(RuntimeError):
    pass


class MangaTypesettingService:

    def __init__(
        self,
        font_path: Optional[str] = None,
        padding_ratio: float = 0.14,
        line_spacing_ratio: float = 0.20,
        all_caps: bool = True
    ):
        self.font_path = font_path or DEFAULT_FONT_PATH
        self.padding_ratio = padding_ratio
        self.line_spacing_ratio = line_spacing_ratio
        self.all_caps = all_caps
        self._font_cache = {}

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
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
        stroke_width: int = 0
    ) -> List[str]:
        words = text.strip().split()
        if not words:
            return []

        lines: List[str] = []
        current_line: List[str] = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font, stroke_width=stroke_width)
            line_w = bbox[2] - bbox[0]

            if line_w <= max_width:
                current_line.append(word)
            elif not current_line:
                lines.append(word)
                current_line = []
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        if len(lines) >= 2 and len(lines[-1].split()) == 1:
            prev_words = lines[-2].split()
            if len(prev_words) >= 3:
                moved_word = prev_words.pop()
                candidate_last_line = f"{moved_word} {lines[-1]}"
                candidate_bbox = draw.textbbox(
                    (0, 0), candidate_last_line, font=font, stroke_width=stroke_width
                )
                candidate_width = candidate_bbox[2] - candidate_bbox[0]
                if candidate_width <= max_width:
                    lines[-2] = " ".join(prev_words)
                    lines[-1] = candidate_last_line

        return lines

    def _measure_text_block(
        self,
        lines: List[str],
        font: ImageFont.ImageFont,
        draw: ImageDraw.ImageDraw,
        stroke_width: int = 0
    ) -> Tuple[int, int, List[int]]:
        if not lines:
            return 0, 0, []

        line_heights = []
        max_w = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            max_w = max(max_w, w)
            line_heights.append(max(h, font.size))

        spacing = max(2, int(round(font.size * self.line_spacing_ratio)))
        total_h = sum(line_heights) + max(0, len(lines) - 1) * spacing
        return max_w, total_h, line_heights

    def find_optimal_font_and_lines(
        self,
        text: str,
        target_w: int,
        target_h: int,
        draw: ImageDraw.ImageDraw,
        page_scale: float = 1.0,
        font_scale: float = 1.0,
        min_font_size: int = 12
    ) -> Tuple[ImageFont.ImageFont, List[str], List[int], int]:
        MIN_FONT_SIZE = 12
        words_count = len(text.strip().split())
        if len(text.strip()) <= 2:
            base_cap = 24
        elif words_count >= 15:
            base_cap = 22
        elif words_count >= 8:
            base_cap = 26
        elif words_count >= 5:
            base_cap = 30
        else:
            base_cap = 38
        page_cap = max(MIN_FONT_SIZE, int(round(base_cap * page_scale * font_scale)))
        width_cap = max(MIN_FONT_SIZE, int(round(max(20 * page_scale, target_w * 0.45) * font_scale)))
        max_size = max(MIN_FONT_SIZE, min(target_h, page_cap, width_cap))

        best_font = None
        best_lines = []
        best_line_heights = []
        best_spacing = 0

        low = max(MIN_FONT_SIZE, min_font_size)
        high = max_size

        while low <= high:
            mid = (low + high) // 2
            test_font = self._get_font(mid)
            stroke_width = max(2, int(round(test_font.size * 0.08)))
            lines = self._wrap_text(
                text, test_font, target_w, draw, stroke_width=stroke_width
            )
            w, h, line_heights = self._measure_text_block(lines, test_font, draw, stroke_width=stroke_width)

            if w <= target_w and h <= target_h:
                best_font = test_font
                best_lines = lines
                best_line_heights = line_heights
                best_spacing = max(2, int(round(test_font.size * self.line_spacing_ratio)))
                low = mid + 1
            else:
                high = mid - 1

        if best_font is None:
            best_font = self._get_font(max(MIN_FONT_SIZE, min_font_size))
            stroke_width = max(2, int(round(best_font.size * 0.08)))
            best_lines = self._wrap_text(
                text, best_font, target_w, draw, stroke_width=stroke_width
            )
            _, _, best_line_heights = self._measure_text_block(
                best_lines, best_font, draw, stroke_width=stroke_width
            )
            best_spacing = max(2, int(round(best_font.size * self.line_spacing_ratio)))

        return best_font, best_lines, best_line_heights, best_spacing

    def typeset(
        self,
        image: Image.Image,
        bubbles: List[DetectedBubble],
        font_scale: float = 1.0,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        stroke_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        if not bubbles or image.width == 0 or image.height == 0:
            return image

        output_img = image.copy()
        overlay = Image.new("RGBA", output_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        page_scale = max(0.5, min(output_img.width / 900.0, output_img.height / 1200.0))
        MIN_FONT_SIZE = 12
        hard_min_font = max(MIN_FONT_SIZE, int(round(14 * page_scale * font_scale)))

        for bubble in bubbles:
            text = bubble.translation or bubble.text
            if not text or not text.strip():
                continue

            if self.all_caps:
                text = text.upper()

            words = text.strip().split()
            bb = bubble.bounding_box
            lb = bubble.layout_box
            tb = bubble.text_box

            if bb:
                center_x = bb.x + bb.width / 2.0
                center_y = bb.y + bb.height / 2.0
                bubble_w = int(bb.width * 0.88)
                bubble_h = int(bb.height * 0.90)
            else:
                fallback_box = lb or tb
                if fallback_box:
                    center_x = fallback_box.x + fallback_box.width / 2.0
                    center_y = fallback_box.y + fallback_box.height / 2.0
                else:
                    center_x = output_img.width / 2.0
                    center_y = output_img.height / 2.0
                bubble_w = 0
                bubble_h = 0

            box_w = int(lb.width * 0.95) if lb else (int(tb.width * 0.95) if tb else 0)
            box_h = int(lb.height * 0.95) if lb else (int(tb.height * 0.95) if tb else 0)

            avail_w = max(24, max(box_w, bubble_w))
            avail_h = max(24, max(box_h, bubble_h))

            if avail_h > avail_w * 1.3:
                avail_w = max(avail_w, min(int(max(avail_w * 1.25, avail_h * 0.50)), output_img.width - 32))

            font, lines, line_heights, spacing = self.find_optimal_font_and_lines(
                text=text,
                target_w=avail_w,
                target_h=avail_h,
                draw=draw,
                page_scale=page_scale,
                font_scale=font_scale,
                min_font_size=hard_min_font
            )

            stroke_width = max(2, int(round(font.size * 0.08))) if stroke_color else 0
            text_w, text_h, _ = self._measure_text_block(lines, font, draw, stroke_width=stroke_width)

            if text_w > avail_w or text_h > avail_h:
                f_try = self._get_font(hard_min_font)
                sw_t = max(2, int(round(f_try.size * 0.08))) if stroke_color else 0
                max_w = max(
                    draw.textbbox((0, 0), w_item, font=f_try, stroke_width=sw_t)[2] -
                    draw.textbbox((0, 0), w_item, font=f_try, stroke_width=sw_t)[0]
                    for w_item in words
                )
                needed_w = max(avail_w, max_w + 14)
                exp_w = min(output_img.width - 32, max(needed_w, int(avail_w * 1.20)))
                exp_h = min(output_img.height - 32, max(avail_h, int(avail_h * 1.25)))

                font_exp, lines_exp, heights_exp, sp_exp = self.find_optimal_font_and_lines(
                    text=text,
                    target_w=exp_w,
                    target_h=exp_h,
                    draw=draw,
                    page_scale=page_scale,
                    font_scale=font_scale,
                    min_font_size=hard_min_font
                )
                sw_exp = max(2, int(round(font_exp.size * 0.08))) if stroke_color else 0
                tw_exp, th_exp, _ = self._measure_text_block(lines_exp, font_exp, draw, stroke_width=sw_exp)

                font, lines, line_heights, spacing = font_exp, lines_exp, heights_exp, sp_exp
                stroke_width = sw_exp
                text_w, text_h = tw_exp, th_exp

            if not lines:
                continue

            box_x = int(round(center_x - text_w / 2.0))
            box_y = int(round(center_y - text_h / 2.0))

            box_x = max(4, min(box_x, output_img.width - text_w - 4))
            box_y = max(4, min(box_y, output_img.height - text_h - 4))

            cur_y = box_y
            for i, line in enumerate(lines):
                line_bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
                lw = line_bbox[2] - line_bbox[0]
                line_x = box_x + (text_w - lw) / 2.0 - line_bbox[0]
                line_y = cur_y - line_bbox[1]

                draw.text(
                    (line_x, line_y),
                    line,
                    font=font,
                    fill=text_color,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color if stroke_width > 0 else None
                )
                cur_y += line_heights[i] + spacing

        ov_arr = np.array(overlay)
        has_alpha = ov_arr[:, :, 3] > 0
        if np.any(has_alpha):
            bg_arr = np.array(output_img)
            if text_color == (0, 0, 0) and stroke_color == (255, 255, 255):
                is_text = (ov_arr[:, :, 0] < 100) & has_alpha
                is_stroke = (ov_arr[:, :, 0] >= 100) & has_alpha
                dest_gray = (
                    0.299 * bg_arr[:, :, 0] + 0.587 * bg_arr[:, :, 1] + 0.114 * bg_arr[:, :, 2]
                )
                is_dark_line = dest_gray < 75
                apply_stroke = is_stroke & (~is_dark_line)

                for c in range(3):
                    bg_arr[:, :, c] = np.where(apply_stroke, ov_arr[:, :, c], bg_arr[:, :, c])
                    bg_arr[:, :, c] = np.where(is_text, ov_arr[:, :, c], bg_arr[:, :, c])

                output_img = Image.fromarray(bg_arr)
            else:
                output_img.paste(overlay, (0, 0), overlay)

        return output_img

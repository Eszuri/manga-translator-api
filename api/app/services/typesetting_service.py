import os
import math
import unicodedata
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from app.schemas import DetectedBubble
from app.core.image_utils import to_rgb_image


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
                # Keep oversized words intact so font fitting must shrink the
                # font instead of accepting a larger font with broken words.
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

    @staticmethod
    def _preferred_font_size(text: str, page_scale: float, font_scale: float, limit: int) -> int:
        base = 30 if len(text.strip()) <= 2 else (32 if len(text.split()) >= 15 else 38)
        return max(12, int(round(min(limit, base * page_scale * font_scale))))

    def find_optimal_font_and_lines(
        self,
        text: str,
        target_w: int,
        target_h: int,
        draw: ImageDraw.ImageDraw,
        page_scale: float = 1.0,
        font_scale: float = 1.0,
        min_font_size: int = 12,
        stroke_enabled: bool = True,
        max_font_size: Optional[int] = None
    ) -> Tuple[ImageFont.ImageFont, List[str], List[int], int]:
        MIN_FONT_SIZE = 12
        if not math.isfinite(font_scale) or font_scale <= 0:
            raise TypesettingError("Font scale must be a finite number greater than zero.")
        max_size = self._preferred_font_size(text, page_scale, font_scale, target_h)
        if max_font_size is not None:
            max_size = min(max_size, max_font_size)

        best_font = None
        best_lines = []
        best_line_heights = []
        best_spacing = 0

        low = max(MIN_FONT_SIZE, min_font_size)
        high = max_size

        while low <= high:
            mid = (low + high) // 2
            test_font = self._get_font(mid)
            stroke_width = max(2, int(round(test_font.size * 0.08))) if stroke_enabled else 0
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
            raise TypesettingError(
                f"Translated text does not fit in the safe area ({target_w} x {target_h} pixels, "
                f"minimum font size {max(MIN_FONT_SIZE, min_font_size)})."
            )

        return best_font, best_lines, best_line_heights, best_spacing

    def _readable_layout(self, text, box, image_size, draw, page_scale,
                         font_scale, stroke_enabled, obstacles,
                         position_guides=None, all_candidates=False, preferred_cap=None):
        page_w, page_h = image_size
        preferred = self._preferred_font_size(
            text, page_scale, font_scale, min(page_w, page_h)
        )
        minimum = max(12, int(round(min(preferred, 14 * page_scale * font_scale))))
        x1, y1, x2, y2 = box
        original_w, original_h = x2 - x1, y2 - y1
        # Derive the font target from local capacity and translated length.
        # A page-wide font target alone made long dialogue spill out even when
        # a smaller, readable font could fit the balloon.
        glyph_count = max(1, sum(not char.isspace() for char in text))
        local_size = math.sqrt(original_w * original_h / (glyph_count * 0.8)) * font_scale
        preferred = min(preferred, max(minimum, int(round(local_size))))
        if preferred_cap is not None:
            preferred = min(preferred, max(minimum, int(round(preferred_cap))))
        sizes = sorted({minimum, *(max(minimum, int(round(preferred * ratio)))
                                  for ratio in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5))},
                       reverse=True)
        fitted_blocks = 0
        candidates = []

        # Allow a modest spill around narrow vertical dialogue, not a new
        # page-wide caption. Only the longest word at the readable floor may
        # exceed this width budget; whole words must never be split.
        floor_font = self._get_font(minimum)
        floor_stroke = max(2, int(round(minimum * 0.08))) if stroke_enabled else 0
        floor_boxes = [draw.textbbox((0, 0), word, font=floor_font,
                                    stroke_width=floor_stroke) for word in text.split()]
        floor_word_width = max(bounds[2] - bounds[0] for bounds in floor_boxes)
        max_width = min(page_w, int(math.ceil(max(
            original_w * 1.6, min(240 * page_scale, original_h * 0.85),
            floor_word_width,
        ))))
        max_height = min(page_h, int(math.ceil(max(original_h * 1.3, preferred * 3))))

        # Evaluate compactness as well as font size. Previously the first font
        # with any page-sized placement won, even across several art panels.
        for size in sizes:
            font = self._get_font(size)
            stroke = max(2, int(round(font.size * 0.08))) if stroke_enabled else 0
            word_boxes = [draw.textbbox((0, 0), word, font=font, stroke_width=stroke)
                          for word in set(text.split())]
            longest = max(bounds[2] - bounds[0] for bounds in word_boxes)
            required_w = max(original_w, longest)
            if longest > max_width:
                continue
            widths = sorted({longest, min(original_w, max_width), max_width,
                             *(min(max_width, max(longest, int(round(original_w * ratio))))
                               for ratio in (0.5, 0.75)),
                             *(min(max_width, int(math.ceil(required_w * ratio)))
                               for ratio in (1.0, 1.25, 1.5, 2.0))})
            for width in widths:
                try:
                    fitted_font, lines, heights, spacing = self.find_optimal_font_and_lines(
                        text, width, max_height, draw, page_scale=page_scale,
                        font_scale=font_scale, min_font_size=size, max_font_size=size,
                        stroke_enabled=stroke_enabled,
                    )
                except TypesettingError:
                    continue
                fitted_blocks += 1
                text_w, text_h, _ = self._measure_text_block(lines, fitted_font, draw, stroke)
                gap = max(2, int(round(size * 0.20)))
                placements = self._place_text_blocks(
                    box, image_size, (text_w, text_h), obstacles, gap, page_scale,
                    position_guides=position_guides,
                )
                for bounds, movement in placements:
                    result = (fitted_font, lines, heights, spacing, bounds)
                    overflow_x = max(0, text_w / original_w - 1)
                    overflow_y = max(0, text_h / original_h - 1)
                    score = (2 * math.log(preferred / size) ** 2
                             + 0.5 * (overflow_x ** 2 + overflow_y ** 2)
                             + 2 * movement / max(original_w, original_h) ** 2)
                    candidates.append(((score, -size, movement), result))
        if candidates:
            if not all_candidates:
                return min(candidates, key=lambda candidate: candidate[0])[1]
            # Keep alternatives at every font size, rather than letting many
            # nearly identical large layouts crowd smaller feasible ones out.
            unique, per_size, results = set(), {}, []
            for _, result in sorted(candidates, key=lambda candidate: candidate[0]):
                font, lines, _, _, bounds = result
                key = (font.size, tuple(lines), bounds)
                if key in unique or per_size.get(font.size, 0) >= 16:
                    continue
                unique.add(key)
                per_size[font.size] = per_size.get(font.size, 0) + 1
                results.append(result)
            return results
        if not fitted_blocks:
            raise TypesettingError(
                f"Whole-word text cannot fit in the local {max_width} x {max_height} pixel area "
                f"at readable font sizes {minimum}-{preferred}."
            )
        raise TypesettingError(
            f"No non-overlapping readable text placement near anchor {box} on the "
            f"{page_w} x {page_h} pixel page; tried reflowing and shifting text "
            f"at font sizes {minimum}-{preferred} against {len(obstacles)} text regions."
        )

    @staticmethod
    def _place_text_block(box, image_size, text_size, obstacles, gap, page_scale):
        placements = MangaTypesettingService._place_text_blocks(
            box, image_size, text_size, obstacles, gap, page_scale
        )
        return placements[0] if placements else None

    @staticmethod
    def _place_text_blocks(box, image_size, text_size, obstacles, gap, page_scale,
                           position_guides=None):
        page_w, page_h = image_size
        text_w, text_h = text_size
        if text_w <= 0 or text_h <= 0 or text_w > page_w or text_h > page_h:
            return []
        x1, y1, x2, y2 = box
        # Start from the nearest on-page position. The mandatory edge correction
        # is not an optional shift toward a different dialogue or panel.
        center_left = max(0, min(page_w - text_w, (x1 + x2 - text_w) / 2))
        center_top = max(0, min(page_h - text_h, (y1 + y2 - text_h) / 2))
        # Narrow Japanese columns need horizontal room for whole Latin words.
        # Permit moving the overflow portion while keeping the block attached
        # to its own anchor; the layout dimensions remain locally bounded.
        shift_x = max((x2 - x1) * 0.25, (text_w - (x2 - x1)) / 2, 12 * page_scale)
        shift_y = max((y2 - y1) * 0.25, (text_h - (y2 - y1)) / 2, 12 * page_scale)
        positions = {(center_left, center_top), (x1, center_top),
                     (x2 - text_w, center_top), (center_left, y1),
                     (center_left, y2 - text_h)}
        # Endpoints are useful even when a neighbour also has to move. Only
        # looking next to its original box cannot discover such joint layouts.
        positions.update((center_left + dx, center_top + dy)
                         for dx in (-shift_x, 0, shift_x)
                         for dy in (-shift_y, 0, shift_y))
        for other in (*obstacles, *(position_guides or ())):
            if (other[2] + gap < center_left - shift_x
                    or other[0] - gap > center_left + shift_x + text_w
                    or other[3] + gap < center_top - shift_y
                    or other[1] - gap > center_top + shift_y + text_h):
                continue
            lefts = (other[0] - gap - text_w, other[2] + gap)
            tops = (other[1] - gap - text_h, other[3] + gap)
            positions.update((left, center_top) for left in lefts)
            positions.update((center_left, top) for top in tops)
            positions.update((left, top) for left in lefts for top in tops)

        # Shifts stay local and intersect their own anchor, so a translation
        # cannot jump to an unrelated empty panel merely to avoid a collision.
        local = set()
        for left, top in positions:
            left = max(0, min(page_w - text_w, int(round(left))))
            top = max(0, min(page_h - text_h, int(round(top))))
            if (abs(left - center_left) > shift_x or abs(top - center_top) > shift_y
                    or left >= x2 or left + text_w <= x1
                    or top >= y2 or top + text_h <= y1):
                continue
            local.add((left, top))
        placements = []
        for left, top in sorted(local, key=lambda position: (
                (position[0] - center_left) ** 2 + (position[1] - center_top) ** 2,
                position[1], position[0])):
            bounds = (left, top, left + text_w, top + text_h)
            if any(bounds[0] < other[2] + gap and bounds[2] > other[0] - gap
                   and bounds[1] < other[3] + gap and bounds[3] > other[1] - gap
                   for other in obstacles):
                continue
            movement = (left - center_left) ** 2 + (top - center_top) ** 2
            placements.append((bounds, movement))
        return placements

    def _plan_layouts(self, entries, image_size, draw, page_scale, font_scale, stroke_enabled,
                      font_caps=None):
        choices = {}
        for index, (bubble_id, text, box) in enumerate(entries):
            try:
                choices[index] = self._readable_layout(
                    text, box, image_size, draw, page_scale, font_scale,
                    stroke_enabled, [], position_guides=[entry[2] for other_index, entry in enumerate(entries)
                                                        if other_index != index],
                    all_candidates=True,
                    preferred_cap=(font_caps or {}).get(index),
                )
            except TypesettingError as exc:
                raise TypesettingError(f"Bubble #{bubble_id}: {exc}") from exc

        def collides(first, second):
            a, b = first[4], second[4]
            gap = max(2, round(max(first[0].size, second[0].size) * 0.20))
            return (a[0] < b[2] + gap and a[2] > b[0] - gap
                    and a[1] < b[3] + gap and a[3] > b[1] - gap)

        # Plan before painting. Revisit earlier choices when a later dialogue
        # is blocked instead of aborting after committing a greedy placement.
        planned = {}
        remaining_budget = 4000
        blocked_index = 0

        def search(remaining):
            nonlocal remaining_budget, blocked_index
            if not remaining:
                return True
            if remaining_budget <= 0:
                return False
            index = min(remaining, key=lambda key: (len(remaining[key]), key))
            blocked_index = index
            for candidate in remaining[index]:
                if remaining_budget <= 0:
                    break
                remaining_budget -= 1
                filtered = {key: [option for option in options if not collides(candidate, option)]
                            for key, options in remaining.items() if key != index}
                if any(not options for options in filtered.values()):
                    continue
                planned[index] = candidate
                if search(filtered):
                    return True
                planned.pop(index, None)
                if remaining_budget <= 0:
                    break
            return False

        if search(choices):
            return [planned[index] for index in range(len(entries))]
        bubble_id, _, box = entries[blocked_index]
        reason = "search limit reached" if remaining_budget <= 0 else "local layout candidates conflict"
        raise TypesettingError(
            f"Cannot place all {len(entries)} dialogues without overlap: {reason}; "
            f"bubble #{bubble_id}, anchor {box}. No partial image was rendered."
        )

    @staticmethod
    def _normalize_render_text(text: str) -> str:
        # The comic font lacks CJK punctuation and full-width Latin glyphs.
        # Keep the translation itself unchanged; use equivalent render glyphs.
        text = unicodedata.normalize("NFKC", text)
        return text.translate(str.maketrans({
            "\u3010": "[", "\u3011": "]", "\u3014": "[", "\u3015": "]",
            "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"',
            "\u3008": "<", "\u3009": ">", "\u300a": "<<", "\u300b": ">>",
            "\u3001": ",", "\u3002": ".", "\u200b": "", "\ufeff": "",
        }))

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

        if not math.isfinite(font_scale) or font_scale <= 0:
            raise TypesettingError("Font scale must be a finite number greater than zero.")

        output_img = to_rgb_image(image).copy()
        # A short page strip uses the same text scale as a full-width page.
        page_scale = max(0.5, output_img.width / 900.0)
        entries = []
        font_caps = {}

        for bubble in bubbles:
            text = bubble.translation or bubble.text
            if not text or not text.strip():
                continue

            if self.all_caps:
                text = text.upper()
            text = self._normalize_render_text(text)
            if not text.strip():
                continue

            # Keep overflow attached to the original dialogue. An estimated
            # empty balloon/layout rectangle can extend toward another bubble.
            box = bubble.text_box or bubble.layout_box or bubble.bounding_box
            layout = bubble.layout_box
            if (layout is not None and layout.x <= box.center_x <= layout.right
                    and layout.y <= box.center_y <= layout.bottom):
                # Use the balloon's inner space only when it does not also
                # contain another dialogue's original glyphs.
                overlaps_other = False
                for other in bubbles:
                    if other is bubble or not (other.translation or other.text or '').strip():
                        continue
                    other_box = other.text_box or other.bounding_box
                    if (layout.x < other_box.right and layout.right > other_box.x
                            and layout.y < other_box.bottom and layout.bottom > other_box.y):
                        overlaps_other = True
                        break
                if not overlaps_other:
                    box = layout
            x1, y1 = max(0, box.x), max(0, box.y)
            x2, y2 = min(output_img.width, box.right), min(output_img.height, box.bottom)
            if x2 <= x1 or y2 <= y1:
                raise TypesettingError(f"Bubble #{bubble.id}: safe area is outside the image.")

            source_count = sum(not char.isspace() for char in (bubble.text or ''))
            if source_count and bubble.text_box is not None:
                # Japanese glyph envelopes provide a local scale even on very
                # wide pages. A short reply must not inherit a page-sized font.
                font_caps[len(entries)] = (
                    math.sqrt(bubble.text_box.area / source_count) * 1.15 * font_scale
                )
            entries.append((bubble.id, text, (x1, y1, x2, y2)))

        layouts = self._plan_layouts(entries, output_img.size, ImageDraw.Draw(output_img),
                                     page_scale, font_scale, stroke_color is not None, font_caps=font_caps)
        for font, lines, line_heights, spacing, bounds in layouts:

            stroke_width = max(2, int(round(font.size * 0.08))) if stroke_color is not None else 0
            left, top, right, bottom = bounds
            text_w, text_h = right - left, bottom - top
            overlay = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            cur_y = 0
            for i, line in enumerate(lines):
                line_bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
                lw = line_bbox[2] - line_bbox[0]
                line_x = (text_w - lw) / 2.0 - line_bbox[0]
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

            # Alpha compositing keeps the white outline visible on dark balloons.
            output_img.paste(overlay, (left, top), overlay)
        return output_img

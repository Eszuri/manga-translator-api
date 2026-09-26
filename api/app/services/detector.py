from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Optional, Dict
from PIL import Image
import numpy as np
import cv2

from app.schemas import BoundingBox, DetectedBubble


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x_left = max(box_a.x, box_b.x)
    y_top = max(box_a.y, box_b.y)
    x_right = min(box_a.right, box_b.right)
    y_bottom = min(box_a.bottom, box_b.bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = box_a.area
    box_b_area = box_b.area
    union_area = float(box_a_area + box_b_area - intersection_area)

    return intersection_area / union_area if union_area > 0 else 0.0


def containment_ratio(inner: BoundingBox, outer: BoundingBox) -> float:
    """Calculate how much of the inner box is contained within the outer box."""
    x_left = max(inner.x, outer.x)
    y_top = max(inner.y, outer.y)
    x_right = min(inner.right, outer.right)
    y_bottom = min(inner.bottom, outer.bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_area = inner.area
    return float(intersection_area) / float(inner_area) if inner_area > 0 else 0.0


def non_max_suppression(
    boxes: List[BoundingBox],
    iou_threshold: float = 0.35,
    containment_threshold: float = 0.70
) -> List[BoundingBox]:
    """Prune redundant/contained overlapping boxes."""
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
    selected: List[BoundingBox] = []

    for box in sorted_boxes:
        should_keep = True
        for kept in selected:
            if calculate_iou(box, kept) > iou_threshold or containment_ratio(box, kept) > containment_threshold:
                should_keep = False
                break
        if should_keep:
            selected.append(box)

    return selected


def merge_adjacent_bubble_boxes(
    boxes: List[Tuple[BoundingBox, str]],
    scale: float = 1.0
) -> List[Tuple[BoundingBox, str]]:
    """
    Merge adjacent text columns/lines that belong to the same multi-column dialogue bubble.
    e.g., column 'じゃあ' next to column '次行くか！' inside the same speech balloon.
    """
    if len(boxes) <= 1:
        return boxes

    merged: List[Tuple[BoundingBox, str]] = []
    used = set()

    for i, (b1, d1) in enumerate(boxes):
        if i in used:
            continue
        group = [b1]
        used.add(i)

        changed = True
        while changed:
            changed = False
            for j, (b2, d2) in enumerate(boxes):
                if j in used or d1 != d2:
                    continue

                for gb in group:
                    dx = max(0, max(gb.x - b2.right, b2.x - gb.right))
                    dy = max(0, max(gb.y - b2.bottom, b2.y - gb.bottom))
                    overlap_y = min(gb.bottom, b2.bottom) - max(gb.y, b2.y)
                    overlap_x = min(gb.right, b2.right) - max(gb.x, b2.x)

                    if d1 == "vertical":
                        # Two vertical dialogue columns side-by-side
                        if dx <= int(52 * scale) and overlap_y >= int(20 * scale):
                            group.append(b2)
                            used.add(j)
                            changed = True
                            break
                    else:
                        # Two horizontal lines stacked vertically
                        if dy <= int(40 * scale) and overlap_x >= int(20 * scale):
                            group.append(b2)
                            used.add(j)
                            changed = True
                            break

        min_x = min(b.x for b in group)
        min_y = min(b.y for b in group)
        max_x = max(b.right for b in group)
        max_y = max(b.bottom for b in group)

        merged.append((
            BoundingBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y),
            d1
        ))

    return merged


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

    if reading_direction == "ltr":
        sorted_list = sorted(bubbles, key=lambda b: (b.bounding_box.y // 60, b.bounding_box.x))
    else:
        avg_height = sum(b.bounding_box.height for b in bubbles) / len(bubbles)
        band_size = max(50.0, avg_height * (1.0 + row_tolerance_ratio))

        def sort_key(item: DetectedBubble) -> Tuple[int, float]:
            band_idx = int(item.bounding_box.center_y // band_size)
            return (band_idx, -item.bounding_box.center_x)

        sorted_list = sorted(bubbles, key=sort_key)

    for idx, bubble in enumerate(sorted_list, start=1):
        bubble.id = idx

    return sorted_list


class BaseBubbleDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> List[DetectedBubble]:
        """Detect speech bubbles from a PIL Image."""
        pass


class ContourBubbleDetector(BaseBubbleDetector):
    """
    High-precision Manga Speech Bubble & Text Detector.

    Features:
    1. Narration Box Extraction: Detects rectangular location/narrator boxes as
       single unified entities without slicing across multi-line horizontal text.
    2. Text-Driven Bubble Extraction: Uses stroke/glyph connected components and
       column/row alignment to detect Japanese dialogue text.
    3. False-Positive Elimination: Uses background purity and 4-way white margin
       checking to reject non-text artwork (faces, jaws, hair, mouths, grass, scenery).
    4. Multi-Column Bubble Merging: Automatically combines adjacent columns and vertically
       stacked text segments belonging to the same speech balloon.
    5. Enclosed Bubble Detection: Detects short dialogue and punctuation bubbles (e.g. 'え…？').
    6. Synthetic Fallback: Preserves support for empty template speech bubbles in unit tests.
    """
    def __init__(
        self,
        min_area_ratio: float = 0.0015,
        max_area_ratio: float = 0.18,
        luminance_threshold: int = 215
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.luminance_threshold = luminance_threshold

    def _detect_narration_boxes(self, gray: np.ndarray, w: int, h: int, scale: float) -> List[Tuple[BoundingBox, str]]:
        """Detect rectangular narration, title, and location banner boxes containing verified text."""
        total_area = h * w
        edges = cv2.Canny(gray, 50, 150)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(edges_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        results: List[Tuple[BoundingBox, str]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < (total_area * 0.001) or area > (total_area * 0.12):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > w * 0.75 or bh > h * 0.35 or bh < 25 or bw < 35:
                continue

            ar = bw / float(bh)
            # Narration banners are distinctly rectangular (not square door/wall frames)
            if not (ar >= 1.35 or ar <= 0.75):
                continue

            hull = cv2.convexHull(c)
            solidity = area / max(cv2.contourArea(hull), 1.0)
            extent = area / max(bw * bh, 1.0)

            # Narration boxes are strictly rectangular
            if extent > 0.85 and solidity > 0.92:
                crop = gray[by:by+bh, bx:bx+bw]
                pad_b = max(4, int(5 * scale))
                inner = crop[pad_b:-pad_b, pad_b:-pad_b]
                if inner.size == 0:
                    continue

                _, ink_inner = cv2.threshold(inner, 120, 255, cv2.THRESH_BINARY_INV)
                nc, _, stats_c, _ = cv2.connectedComponentsWithStats(ink_inner)

                box_chars = []
                for i in range(1, nc):
                    cx, cy, cw, ch, ca = stats_c[i]
                    if 6 <= cw <= 60 and 6 <= ch <= 60 and ca >= 20:
                        box_chars.append((cx, cy, cw, ch, ca))

                has_line = False
                # Vertical check
                chars_y = sorted(box_chars, key=lambda c: (c[1], c[0]))
                for i, c1 in enumerate(chars_y):
                    line = [c1]; curr = c1
                    for j in range(i+1, len(chars_y)):
                        c2 = chars_y[j]
                        dx = abs((curr[0]+curr[2]/2.0) - (c2[0]+c2[2]/2.0))
                        dy = c2[1] - (curr[1]+curr[3])
                        if dx <= max(curr[2], c2[2]) * 0.35 and -5 <= dy <= max(curr[3], c2[3]) * 1.2:
                            line.append(c2); curr = c2
                    if len(line) >= 3:
                        has_line = True; break

                # Horizontal check
                if not has_line:
                    chars_x = sorted(box_chars, key=lambda c: (c[0], c[1]))
                    for i, c1 in enumerate(chars_x):
                        line = [c1]; curr = c1
                        for j in range(i+1, len(chars_x)):
                            c2 = chars_x[j]
                            dy = abs((curr[1]+curr[3]/2.0) - (c2[1]+c2[3]/2.0))
                            dx = c2[0] - (curr[0]+curr[2])
                            if dy <= max(curr[3], c2[3]) * 0.35 and -5 <= dx <= max(curr[2], c2[2]) * 1.2:
                                line.append(c2); curr = c2
                        if len(line) >= 3:
                            has_line = True; break

                if has_line:
                    direction = "horizontal" if ar >= 1.2 else "vertical"
                    results.append((BoundingBox(x=bx, y=by, width=bw, height=bh), direction))

        return results

    def _detect_text_bubbles(self, gray: np.ndarray, w: int, h: int, scale: float) -> List[Tuple[BoundingBox, str]]:
        """Detect dialogue speech bubbles anchored on primary Japanese text columns with generous balloon margins."""
        # 1. Extract dark ink glyphs
        _, ink = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ink_dilated = cv2.dilate(ink, k2)
        num_c, _, stats_c, _ = cv2.connectedComponentsWithStats(ink_dilated)

        chars = []
        for i in range(1, num_c):
            cx, cy, cw, ch, ca = stats_c[i]
            min_w = int(6 * scale)
            min_h = int(14 * scale)
            max_dim = int(60 * scale)

            if min_w <= cw <= max_dim and min_h <= ch <= max_dim and ca >= int(35 * scale):
                ar = cw / float(ch)
                density = ca / float(cw * ch)
                # Allow thin kana marks (ー, ！, furigana) up to wide kanji
                if 0.10 <= ar <= 3.2 and 0.08 <= density <= 0.90:
                    chars.append((cx, cy, cw, ch, ca))

        # 2. Vertical text columns (Tategaki - standard manga dialogue)
        chars_y = sorted(chars, key=lambda c: (c[1], c[0]))
        v_lines = []
        used_v = set()
        for i, c1 in enumerate(chars_y):
            if i in used_v:
                continue
            line = [c1]
            used_v.add(i)
            curr = c1
            while True:
                best_j = None
                best_dist = float('inf')
                font_s = max(curr[2], curr[3])
                for j, c2 in enumerate(chars_y):
                    if j in used_v:
                        continue
                    cx1 = curr[0] + curr[2] / 2.0
                    cx2 = c2[0] + c2[2] / 2.0
                    dx = abs(cx1 - cx2)
                    dy = c2[1] - (curr[1] + curr[3])
                    font_s2 = max(c2[2], c2[3])
                    f_max = max(font_s, font_s2)
                    # Strict centerline alignment eliminates speedlines and faceted artwork
                    if dx <= max(f_max * 0.22, 6 * scale) and -8 * scale <= dy <= max(f_max * 0.95, 30 * scale):
                        dist = dy + dx * 2
                        if dist < best_dist:
                            best_dist = dist
                            best_j = j
                if best_j is not None:
                    line.append(chars_y[best_j])
                    used_v.add(best_j)
                    curr = chars_y[best_j]
                else:
                    break
            v_lines.append(line)

        # 3. Margin & white paper validation
        m = int(14 * scale)
        def check_bubble_margin(l):
            lx = min(c[0] for c in l); ly = min(c[1] for c in l)
            lw = max(c[0]+c[2] for c in l) - lx; lh = max(c[1]+c[3] for c in l) - ly
            if ly <= 10 or ly+lh >= h-10 or lx <= 10 or lx+lw >= w-10:
                return False, None
            l_m = gray[ly:ly+lh, max(0, lx-m):lx]
            r_m = gray[ly:ly+lh, lx+lw:min(w, lx+lw+m)]
            t_m = gray[max(0, ly-m):ly, lx:lx+lw]
            b_m = gray[ly+lh:min(h, ly+lh+m), lx:lx+lw]
            def wr(arr): return np.mean(arr > 200) if arr.size > 0 else 1.0
            wl, wr_, wt, wb = wr(l_m), wr(r_m), wr(t_m), wr(b_m)
            margin_score = sum(1 for val in [wl, wr_, wt, wb] if val >= 0.70)
            crop = gray[max(0, ly-m):min(h, ly+lh+m), max(0, lx-m):min(w, lx+lw+m)]
            bg_pixels = crop[crop > 130]
            bg_white = np.mean(crop > 200)
            bg_std = np.std(bg_pixels) if bg_pixels.size > 0 else 0
            if margin_score >= 3 and bg_white >= 0.75 and bg_std <= 22.5:
                return True, (lx, ly, lw, lh)
            return False, None

        primaries = []
        secondaries = []
        for l in v_lines:
            lx = min(c[0] for c in l); ly = min(c[1] for c in l)
            lw = max(c[0]+c[2] for c in l) - lx; lh = max(c[1]+c[3] for c in l) - ly
            ok, box = check_bubble_margin(l)
            if ok and len(l) >= 3 and sum(c[4] for c in l) / len(l) >= 50 * scale:
                primaries.append(box)
            elif len(l) >= 1:
                secondaries.append((lx, ly, lw, lh))

        def can_merge(b_box: Tuple[int, int, int, int], cand: Tuple[int, int, int, int]) -> bool:
            dx = max(0, max(b_box[0], cand[0]) - min(b_box[0]+b_box[2], cand[0]+cand[2]))
            dy = max(0, max(b_box[1], cand[1]) - min(b_box[1]+b_box[3], cand[1]+cand[3]))
            overlap_y = max(0, min(b_box[1]+b_box[3], cand[1]+cand[3]) - max(b_box[1], cand[1]))
            overlap_x = max(0, min(b_box[0]+b_box[2], cand[0]+cand[2]) - max(b_box[0], cand[0]))
            is_side_by_side = (dx <= int(65 * scale) and overlap_y >= int(8 * scale))
            is_stacked = (dy <= int(45 * scale) and overlap_x >= int(10 * scale))
            is_staggered = (dx <= int(55 * scale) and dy <= int(75 * scale))
            if not (is_side_by_side or is_stacked or is_staggered):
                return False
            bx1 = min(b_box[0], cand[0])
            by1 = min(b_box[1], cand[1])
            bx2 = max(b_box[0]+b_box[2], cand[0]+cand[2])
            by2 = max(b_box[1]+b_box[3], cand[1]+cand[3])
            bridge = gray[by1:by2, bx1:bx2]
            return bridge.size > 0 and np.mean(bridge > 175) >= 0.65

        # 4. Bubble clustering anchored on primary dialogue columns
        bubbles: List[List[Tuple[int, int, int, int]]] = []
        for p in primaries:
            merged_into = None
            for b in bubbles:
                for member in b:
                    if can_merge(member, p):
                        merged_into = b
                        break
                if merged_into:
                    break
            if merged_into:
                merged_into.append(p)
            else:
                bubbles.append([p])

        # Attach secondary short columns (e.g. お兄, 部活, 今日は, きた) belonging to the same bubble
        for s in secondaries:
            best_b = None
            best_score = -1
            for b in bubbles:
                for member in b:
                    overlap_y = max(0, min(member[1]+member[3], s[1]+s[3]) - max(member[1], s[1]))
                    overlap_x = max(0, min(member[0]+member[2], s[0]+s[2]) - max(member[0], s[0]))
                    if can_merge(member, s):
                        score = overlap_y * 2 + overlap_x
                        if score > best_score:
                            best_score = score
                            best_b = b
            if best_b is not None:
                best_b.append(s)

        # 5. Form final bounding boxes fitting speech balloon geometry
        _, white_mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = w * h
        balloon_contours = []
        for c in cnts:
            c_area = cv2.contourArea(c)
            if c_area < total_area * 0.002 or c_area > total_area * 0.15:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            solidity = c_area / max(cv2.contourArea(hull), 1.0)
            if solidity >= 0.58 and bw < w * 0.65 and bh < h * 0.55:
                balloon_contours.append((bx, by, bw, bh))

        merged_bubbles: List[Tuple[BoundingBox, str]] = []
        for b in bubbles:
            gx1 = min(m[0] for m in b)
            gy1 = min(m[1] for m in b)
            gx2 = max(m[0]+m[2] for m in b)
            gy2 = max(m[1]+m[3] for m in b)
            center_x = (gx1 + gx2) / 2.0
            center_y = (gy1 + gy2) / 2.0
            text_w = gx2 - gx1
            text_h = gy2 - gy1

            matched_balloon = None
            for (bx, by, bw, bh) in balloon_contours:
                if bx <= center_x <= bx + bw and by <= center_y <= by + bh:
                    overlap_x = max(0, min(gx2, bx + bw) - max(gx1, bx))
                    overlap_y = max(0, min(gy2, by + bh) - max(gy1, by))
                    if overlap_x >= text_w * 0.80 and overlap_y >= text_h * 0.80:
                        matched_balloon = (bx, by, bw, bh)
                        break

            if matched_balloon:
                fbx, fby, fbw, fbh = matched_balloon
                merged_bubbles.append((BoundingBox(x=fbx, y=fby, width=fbw, height=fbh), "vertical"))
            else:
                pad_x = int(26 * scale)
                pad_y = int(22 * scale)
                fx1 = max(0, gx1 - pad_x)
                fy1 = max(0, gy1 - pad_y)
                fw = min(w, gx2 + pad_x) - fx1
                fh = min(h, gy2 + pad_y) - fy1
                merged_bubbles.append((BoundingBox(x=fx1, y=fy1, width=fw, height=fh), "vertical"))

        # Deduplicate identical or heavily overlapping balloon detections
        deduped: List[Tuple[BoundingBox, str]] = []
        for box, direction in merged_bubbles:
            is_dup = False
            for prev_box, _ in deduped:
                inter_x1 = max(box.x, prev_box.x)
                inter_y1 = max(box.y, prev_box.y)
                inter_x2 = min(box.right, prev_box.right)
                inter_y2 = min(box.bottom, prev_box.bottom)
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                union_area = box.area + prev_box.area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > 0.65:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append((box, direction))

        return deduped

    def _detect_synthetic_fallback(self, gray: np.ndarray, w: int, h: int) -> List[Tuple[BoundingBox, str]]:
        """Fallback for synthetic unit-test manga templates with empty white speech bubbles."""
        total_image_area = h * w
        _, binary = cv2.threshold(gray, self.luminance_threshold, 255, cv2.THRESH_BINARY)
        ksize = max(5, int(min(w, h) * 0.015))
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_pixels = total_image_area * self.min_area_ratio
        max_pixels = total_image_area * self.max_area_ratio
        boxes: List[Tuple[BoundingBox, str]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_pixels <= area <= max_pixels:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = bh / bw if bw > 0 else 1.0
                if 0.15 <= aspect_ratio <= 6.0:
                    dir_str = "vertical" if aspect_ratio >= 1.1 else "horizontal"
                    boxes.append((BoundingBox(x=bx, y=by, width=bw, height=bh), dir_str))

        return boxes

    def detect(self, image: Image.Image) -> List[DetectedBubble]:
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = max(w, h) / 1600.0

        # 1. Detect Rectangular Narration Boxes with Verified Text
        narration_boxes = self._detect_narration_boxes(gray, w, h, scale)

        # 2. Detect Text-Driven Speech Bubbles
        text_bubbles = self._detect_text_bubbles(gray, w, h, scale)

        combined = narration_boxes + text_bubbles

        # 3. Fallback for synthetic/empty bubble manga templates (used in unit tests)
        if not combined:
            combined = self._detect_synthetic_fallback(gray, w, h)

        if not combined:
            return []

        # 5. Non-Maximum Suppression to remove overlapping duplicates
        sorted_boxes = sorted(combined, key=lambda item: item[0].area, reverse=True)
        selected: List[Tuple[BoundingBox, str]] = []
        for item in sorted_boxes:
            box = item[0]
            keep = True
            for kept, _ in selected:
                ix1 = max(box.x, kept.x)
                iy1 = max(box.y, kept.y)
                ix2 = min(box.right, kept.right)
                iy2 = min(box.bottom, kept.bottom)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if (inter / box.area) > 0.65:
                        keep = False
                        break
            if keep:
                selected.append(item)

        results: List[DetectedBubble] = []
        for idx, (box, direction) in enumerate(selected, start=1):
            ratio = round(box.height / box.width, 2)
            results.append(
                DetectedBubble(
                    id=idx,
                    bounding_box=box,
                    confidence=0.96,
                    direction=direction,
                    aspect_ratio=ratio
                )
            )

        return results



class MockBubbleDetector(BaseBubbleDetector):
    """
    Deterministic detector for unit tests and simulation.
    """
    def __init__(self, predefined_bubbles: List[DetectedBubble] = None):
        self.predefined_bubbles = predefined_bubbles

    def detect(self, image: Image.Image) -> List[DetectedBubble]:
        if self.predefined_bubbles is not None:
            return self.predefined_bubbles

        w, h = image.size
        b1 = BoundingBox(x=int(w * 0.60), y=int(h * 0.10), width=int(w * 0.25), height=int(h * 0.25))
        b2 = BoundingBox(x=int(w * 0.15), y=int(h * 0.55), width=int(w * 0.30), height=int(h * 0.20))

        return [
            DetectedBubble(
                id=1,
                bounding_box=b1,
                confidence=0.98,
                direction="vertical",
                aspect_ratio=round(b1.height / b1.width, 2)
            ),
            DetectedBubble(
                id=2,
                bounding_box=b2,
                confidence=0.95,
                direction="horizontal",
                aspect_ratio=round(b2.height / b2.width, 2)
            ),
        ]

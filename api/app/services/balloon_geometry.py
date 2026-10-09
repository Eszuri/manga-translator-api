from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.schemas import BoundingBox


def clip_box(box: BoundingBox, width: int, height: int) -> Optional[BoundingBox]:
    x, y = max(0, box.x), max(0, box.y)
    right, bottom = min(width, box.right), min(height, box.bottom)
    if right <= x or bottom <= y:
        return None
    return BoundingBox(x=x, y=y, width=right - x, height=bottom - y)


def union_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    x, y = min(b.x for b in boxes), min(b.y for b in boxes)
    return BoundingBox(x=x, y=y, width=max(b.right for b in boxes) - x,
                       height=max(b.bottom for b in boxes) - y)


def adjacent_columns(a: BoundingBox, b: BoundingBox) -> bool:
    overlap_y = max(0, min(a.bottom, b.bottom) - max(a.y, b.y))
    overlap_x = max(0, min(a.right, b.right) - max(a.x, b.x))
    if overlap_x * overlap_y > 0.55 * min(a.area, b.area):
        return True
    gap = max(0, max(a.x, b.x) - min(a.right, b.right))
    return (overlap_y >= 0.65 * min(a.height, b.height)
            and max(a.height, b.height) <= 1.8 * min(a.height, b.height)
            and gap <= max(4, 0.5 * min(a.width, b.width)))


def refine_text_boxes(text: BoundingBox, gray: np.ndarray, segmentation: np.ndarray,
                      direction: str) -> List[BoundingBox]:
    original = text
    pad = max(4, min(12, round(min(text.width, text.height) * 0.15)))
    x, y = max(0, text.x - pad), max(0, text.y - pad)
    text = BoundingBox(x=x, y=y, width=min(gray.shape[1], text.right + pad) - x,
                       height=min(gray.shape[0], text.bottom + pad) - y)
    patch = gray[text.y:text.bottom, text.x:text.right]
    support = ((segmentation[text.y:text.bottom, text.x:text.right] > 0.3)
               & (patch < 180)).astype(np.uint8)
    # Refinement may tighten a detector box only when the segmentation covers
    # its ink reliably. A missed short column must not disappear from the OCR
    # crop just because the other, longer column has a strong mask.
    original_ink = gray[original.y:original.bottom, original.x:original.right] < 180
    original_support = segmentation[original.y:original.bottom, original.x:original.right] > 0.3
    ink_count = np.count_nonzero(original_ink)
    if ink_count and np.count_nonzero(original_ink & original_support) < ink_count * 0.75:
        return [original]
    if np.count_nonzero(support) < 12:
        return [original]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    glyphs = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 3]
    if not glyphs:
        return [original]
    sizes = [stats[i, cv2.CC_STAT_HEIGHT] for i in glyphs
             if 4 <= stats[i, cv2.CC_STAT_HEIGHT] <= max(16, text.height * 0.2)]
    char_height = float(np.percentile(sizes, 80)) if sizes else max(4, text.height / 15)
    clean = np.zeros_like(support)
    for i in glyphs:
        _, _, w, h, area = stats[i]
        if h > max(4 * char_height, 0.45 * text.height) or w > max(4 * char_height, 0.85 * text.width):
            continue
        clean[labels == i] = 1
    if np.count_nonzero(clean) < 12:
        return [original]
    if direction != 'vertical':
        yy, xx = np.nonzero(clean)
        return [BoundingBox(x=text.x + int(xx.min()), y=text.y + int(yy.min()),
                            width=int(xx.max() - xx.min() + 1), height=int(yy.max() - yy.min() + 1))]

    row_ink = clean.sum(axis=1)
    quiet = row_ink <= max(2, round(text.width * 0.015))
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
    split = None
    best_score = 0.0
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start < max(8, char_height * 0.45):
            continue
        before, after = clean[:start], clean[stop:]
        if min(before.sum(), after.sum()) < max(50, clean.sum() * 0.12):
            continue
        by, bx = np.nonzero(before)
        ay, ax = np.nonzero(after)
        if min(np.ptp(by), np.ptp(ay)) < 2 * char_height:
            continue
        shift = abs(float(bx.mean() - ax.mean()))
        if shift < text.width * 0.23:
            continue
        score = (stop - start) * shift
        if score > best_score:
            best_score, split = score, (start, stop)
    if split is not None:
        result = []
        for part in (clean[:split[0]], clean[split[1]:]):
            offset_y = 0 if not result else split[1]
            cols = part.sum(axis=0)
            low = cols <= 2
            bounds = np.flatnonzero(np.diff(np.r_[False, low, False]))
            bands = []
            left = 0
            for a, z in zip(bounds[::2], bounds[1::2]):
                if z - a >= char_height * 0.6:
                    bands.append((left, a, int(cols[left:a].sum())))
                    left = z
            bands.append((left, len(cols), int(cols[left:].sum())))
            bands = [band for band in bands if band[2] > 0]
            if len(bands) > 1:
                dominant = max(bands, key=lambda band: band[2])
                if dominant[2] >= 1.5 * max(b[2] for b in bands if b is not dominant):
                    part = part.copy()
                    part[:, :dominant[0]] = 0
                    part[:, dominant[1]:] = 0
            yy, xx = np.nonzero(part)
            result.append(BoundingBox(x=text.x + int(xx.min()), y=text.y + offset_y + int(yy.min()),
                                      width=int(xx.max() - xx.min() + 1),
                                      height=int(yy.max() - yy.min() + 1)))
        return result
    kernel = np.ones((max(3, round(char_height * 1.4)) | 1,
                      max(3, round(char_height * 0.25)) | 1), np.uint8)
    joined = cv2.dilate(clean, kernel)
    count, columns, _, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = []
    for label in range(1, count):
        yy, xx = np.nonzero((columns == label) & (clean > 0))
        if len(xx) < 8:
            continue
        boxes.append(BoundingBox(x=text.x + int(xx.min()), y=text.y + int(yy.min()),
                                  width=int(xx.max() - xx.min() + 1), height=int(yy.max() - yy.min() + 1)))
    groups = []
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                b = boxes[j]
                overlap_x = max(0, min(a.right, b.right) - max(a.x, b.x))
                gap_y = max(0, max(a.y, b.y) - min(a.bottom, b.bottom))
                if overlap_x >= 0.55 * min(a.width, b.width) and gap_y <= 3 * char_height:
                    boxes[i] = union_boxes([a, b])
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    for box in sorted(boxes, key=lambda b: -b.x):
        for group in groups:
            envelope = union_boxes(group)
            overlap = max(0, min(box.bottom, envelope.bottom) - max(box.y, envelope.y))
            gap = max(0, max(box.x, envelope.x) - min(box.right, envelope.right))
            if overlap >= 0.6 * min(box.height, envelope.height) and gap <= 2.2 * char_height:
                group.append(box)
                break
        else:
            groups.append([box])
    envelopes = [union_boxes(g) for g in groups]
    large = [b for b in envelopes if b.height >= max(2 * char_height, text.height * 0.18)]
    if not large:
        return [original]
    small_boxes = [b for b in envelopes if b not in large]
    for small in small_boxes:
        nearest = min(range(len(large)), key=lambda i: (large[i].center_x - small.center_x) ** 2
                      + (large[i].center_y - small.center_y) ** 2)
        large[nearest] = union_boxes([large[nearest], small])
    return large


def interior_polygon(mask: np.ndarray, origin: Tuple[int, int]) -> List[Tuple[int, int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, 1.0, True)
    return [(int(p[0][0]) + origin[0], int(p[0][1]) + origin[1]) for p in contour]


def safe_layout_box(mask: np.ndarray, origin: Tuple[int, int], text: BoundingBox) -> Optional[BoundingBox]:
    h, w = mask.shape
    inset = max(2, round(min(w, h) * 0.035))
    safe = cv2.erode(mask.astype(np.uint8), np.ones((2 * inset + 1, 2 * inset + 1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    step = max(1, int(np.ceil(max(w, h) / 180)))
    gh, gw = h // step, w // step
    if not gh or not gw:
        return None
    grid = safe[:gh * step, :gw * step].reshape(gh, step, gw, step).min(axis=(1, 3)) > 0
    anchor_x = (text.center_x - origin[0]) / step
    anchor_y = (text.center_y - origin[1]) / step
    histogram = np.zeros(gw, dtype=int)
    best, best_score = None, 0.0
    for row in range(gh):
        histogram = (histogram + 1) * grid[row]
        stack = []
        for col in range(gw + 1):
            value = int(histogram[col]) if col < gw else 0
            start = col
            while stack and stack[-1][1] > value:
                left, height = stack.pop()
                top = row + 1 - height
                start = left
                if left <= anchor_x < col and top <= anchor_y < row + 1:
                    rw, rh = col - left, height
                    dx = abs((left + col) / 2 - anchor_x) / max(1, rw)
                    dy = abs((top + row + 1) / 2 - anchor_y) / max(1, rh)
                    score = rw * rh / (1 + dx + dy)
                    if score > best_score:
                        best_score, best = score, (left, top, rw, rh)
            if value and (not stack or stack[-1][1] < value):
                stack.append((start, value))
    if best is None:
        return None
    x, y, bw, bh = best
    return BoundingBox(x=origin[0] + x * step, y=origin[1] + y * step,
                       width=bw * step, height=bh * step)


class BalloonRegions:

    def __init__(self, gray: np.ndarray, white_threshold: int = 195,
                 text_boxes: Optional[List[BoundingBox]] = None):
        self.gray = gray
        self.white_threshold = white_threshold
        self.h, self.w = gray.shape
        self.text_boxes = text_boxes or []
        self.levels = []
        local_scale = min(max(gray.shape), min(gray.shape) * 2)
        ink = (gray <= white_threshold).astype(np.uint8)
        for text in text_boxes or []:
            box = clip_box(text, self.w, self.h)
            if box is not None:
                pad = max(2, min(round(local_scale * 0.002), round(min(box.width, box.height) * 0.1)))
                ink[max(0, box.y - pad):min(self.h, box.bottom + pad),
                    max(0, box.x - pad):min(self.w, box.right + pad)] = 0
        for size in sorted({3, max(3, round(local_scale * 0.005)) | 1,
                            max(5, round(local_scale * 0.011)) | 1}):
            closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
            _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - closed, connectivity=4)
            self.levels.append((labels, stats))

    def match(self, text: BoundingBox):
        best = None
        best_area = float('inf')
        patch_gray = self.gray[text.y:text.bottom, text.x:text.right]
        paper = patch_gray > self.white_threshold
        paper_count = max(1, np.count_nonzero(paper))
        for level, (labels, stats) in enumerate(self.levels):
            patch = labels[text.y:text.bottom, text.x:text.right]
            ids, counts = np.unique(patch[paper], return_counts=True)
            for label, count in zip(ids, counts):
                if not label or count < 0.55 * paper_count:
                    continue
                x, y, w, h, area = map(int, stats[label])
                supported_area = max(text.area, sum(b.area for b in self.text_boxes
                    if 0 <= int(b.center_x) < self.w and 0 <= int(b.center_y) < self.h
                    and labels[int(b.center_y), int(b.center_x)] == label))
                # A short Japanese column can occupy very little of a balloon.
                # Relax the text-area ratio only for a compact, enclosed region
                # found with the smallest closing kernel, not open page paper
                # or regions created by bridging larger gaps in the artwork.
                short_enclosed = (
                    level == 0 and x > 0 and y > 0
                    and x + w < self.w and y + h < self.h
                    and area / (w * h) >= 0.55
                    and text.width <= w * 0.35 and text.height <= h * 0.55
                )
                area_ratio = 64 if short_enclosed else 14
                if (area < 40 or area > self.w * self.h * 0.28
                        or w > self.w * 0.8 or h > self.h * 0.75
                        or w * h > supported_area * area_ratio or area / (w * h) < 0.3):
                    continue
                if not (x <= text.center_x < x + w and y <= text.center_y < y + h):
                    continue
                if area < best_area:
                    best_area = area
                    best = (level, int(label), (x, y, w, h))
        return best

    def mask(self, match):
        level, label, (x, y, w, h) = match
        mask = (self.levels[level][0][y:y + h, x:x + w] == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
        return mask

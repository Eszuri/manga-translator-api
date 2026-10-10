from typing import List, Optional

import numpy as np
from PIL import Image
import cv2

from app.schemas import BoundingBox, DetectedBubble
from app.services.detector import BaseBubbleDetector, sort_manga_reading_order
from app.services.comic_text_detector import ComicTextDetector
from app.services.balloon_geometry import (
    BalloonRegions, adjacent_columns, clip_box, interior_polygon, safe_layout_box, union_boxes,
    split_connected_balloons,
)


class HybridBubbleDetector(BaseBubbleDetector):

    def __init__(self, comic_detector: Optional[ComicTextDetector] = None,
                 conf_threshold: float = 0.35, nms_threshold: float = 0.35,
                 white_threshold: int = 195, num_threads: int = 4):
        self.comic_detector = comic_detector or ComicTextDetector(
            conf_threshold=conf_threshold, nms_threshold=nms_threshold,
            num_threads=num_threads)
        self.white_threshold = white_threshold

    def detect(self, image: Image.Image) -> List[DetectedBubble]:
        raw = self.comic_detector.detect(image)
        if not raw:
            return []
        width, height = image.size
        # A long strip must not raise the minimum area of an unchanged local
        # dialogue merely because more panels were appended below it.
        local_page_area = width * min(height, width * 2)
        gray = np.array(image.convert('L'))
        regions = BalloonRegions(gray, self.white_threshold,
                                 [b.text_box or b.bounding_box for b in raw])
        outline_regions = BalloonRegions(gray, self.white_threshold, outline_only=True)
        segmentation = self.comic_detector.get_cached_segmentation(image)
        groups = []
        for bubble in raw:
            text = clip_box(bubble.text_box or bubble.bounding_box, width, height)
            if text is None:
                continue
            match = regions.match(text)
            if (text.area < max(300, local_page_area * 0.0004)
                    and (match is None or match[2][2] * match[2][3] < max(2500, local_page_area * 0.002))):
                continue
            for group in groups:
                if (match is not None and group['match'] is not None
                        and match[:2] == group['match'][:2]
                        and all(adjacent_columns(text, t) for t in group['texts'])
                        and (segmentation is None or len(split_connected_balloons(
                            union_boxes([text, *group['texts']]), gray, segmentation, outline_regions)) == 1)):
                    group['texts'].append(text)
                    group['members'].append(bubble)
                    break
            else:
                groups.append(dict(texts=[text], members=[bubble], match=match))

        results = []
        for group in groups:
            text = union_boxes(group['texts'])
            match = group['match']
            polygon, layout = None, None
            if match is not None:
                x, y, w, h = match[2]
                mask = regions.mask(match)
                yy, xx = np.mgrid[y:y + h, x:x + w]

                def distance(box):
                    dx = np.maximum(np.maximum(box.x - xx, xx - box.right), 0)
                    dy = np.maximum(np.maximum(box.y - yy, yy - box.bottom), 0)
                    return dx * dx + dy * dy

                own_distance = distance(text)
                for other in groups:
                    if other is group or other['match'] is None:
                        continue
                    other_text = union_boxes(other['texts'])
                    cx, cy = int(other_text.center_x) - x, int(other_text.center_y) - y
                    if 0 <= cx < w and 0 <= cy < h and mask[cy, cx]:
                        mask[distance(other_text) < own_distance] = 0
                polygon = interior_polygon(mask, (x, y))
                layout = safe_layout_box(mask, (x, y), text)
                if polygon:
                    px, py, pw, ph = cv2.boundingRect(np.array(polygon, np.int32))
                    filled = np.zeros((ph, pw), np.uint8)
                    cv2.fillPoly(filled, [np.array([(cx - px, cy - py) for cx, cy in polygon],
                                                       np.int32)], 1)
                    alternate = safe_layout_box(filled, (px, py), text)
                    if alternate and (layout is None or alternate.area > layout.area * 1.15):
                        def overlaps_other_text(other):
                            box = union_boxes(other['texts'])
                            return (max(0, min(alternate.right, box.right) - max(alternate.x, box.x))
                                    * max(0, min(alternate.bottom, box.bottom) - max(alternate.y, box.y)) > 0)

                        overlaps_dialogue = any(other is not group and overlaps_other_text(other)
                                                for other in groups)
                        if not overlaps_dialogue:
                            layout = alternate
                if polygon:
                    xs, ys = zip(*polygon)
                    balloon = BoundingBox(x=min(xs), y=min(ys),
                                          width=max(xs) - min(xs) + 1,
                                          height=max(ys) - min(ys) + 1)
                else:
                    balloon = text
            else:
                pad = max(2, round(min(max(width, height), width * 2) * 0.004))
                x, y = max(0, text.x - pad), max(0, text.y - pad)
                balloon = BoundingBox(x=x, y=y, width=min(width, text.right + pad) - x,
                                      height=min(height, text.bottom + pad) - y)
            if layout is None:
                layout = text
            members = group['members']
            results.append(DetectedBubble(
                id=len(results) + 1, bounding_box=balloon, text_box=text,
                layout_box=layout, bubble_polygon=polygon or None,
                confidence=round(sum(b.confidence for b in members) / len(members), 2),
                direction=members[0].direction,
                aspect_ratio=round(balloon.height / balloon.width, 2)))
        return sort_manga_reading_order(results, reading_direction='rtl')

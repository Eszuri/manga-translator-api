from typing import List, Tuple, Dict, Optional
from PIL import Image
import numpy as np
import cv2

from app.schemas import BoundingBox, DetectedBubble
from app.services.detector import BaseBubbleDetector, sort_manga_reading_order
from app.services.comic_text_detector import ComicTextDetector


class HybridBubbleDetector(BaseBubbleDetector):
    """
    Hybrid Manga Bubble & Text Detector:
    1. AI (ComicTextDetector): Pinpoints exact character text coordinates (text_box).
    2. OpenCV Segmentation: Finds the outer speech balloon boundary (bounding_box) seeded by the text.
    3. Bubble Unification: Groups multiple text columns only if they reside in the same enclosing balloon.
    4. False Positive Rejection: Only candidate bubbles containing confirmed text are retained.
    """

    def __init__(
        self,
        comic_detector: Optional[ComicTextDetector] = None,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.35,
        white_threshold: int = 195,
        num_threads: int = 4,
        require_gpu: bool = False,
        device: str = "auto"
    ):
        self.comic_detector = comic_detector or ComicTextDetector(
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            num_threads=num_threads,
            require_gpu=require_gpu,
            device=device
        )
        self.white_threshold = white_threshold

    def _find_panel_dividers(self, gray: np.ndarray) -> Tuple[List[int], List[int]]:
        """Identifies horizontal and vertical panel gutter lines."""
        h, w = gray.shape
        h_dividers = [
            py for py in range(h)
            if np.mean(gray[py, :] < 65) > 0.40
        ]
        v_dividers = [
            px for px in range(w)
            if np.mean(gray[:, px] < 65) > 0.40
        ]
        return h_dividers, v_dividers

    def detect(self, image: Image.Image) -> List[DetectedBubble]:
        """
        Executes hybrid text and bubble detection.
        Returns DetectedBubble list with distinct bounding_box (balloon) and text_box (text).
        """
        orig_w, orig_h = image.size
        if orig_w == 0 or orig_h == 0:
            return []

        # 1. Step 1: Detect exact text lines using ComicTextDetector
        raw_text_bubbles = self.comic_detector.detect(image)
        if not raw_text_bubbles:
            return []

        # 2. Step 2: Analyze image structure for white balloon regions and panel lines
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        total_area = w * h
        scale = max(w, h) / 1000.0

        h_dividers, v_dividers = self._find_panel_dividers(gray)

        # Segment white paper areas
        white_mask = (gray > self.white_threshold).astype(np.uint8)
        num_white, white_labels, white_stats, _ = cv2.connectedComponentsWithStats(
            white_mask, connectivity=8
        )

        balloon_regions: Dict[int, Tuple[int, int, int, int, int]] = {}
        for label_id in range(1, num_white):
            bx, by, bw, bh, region_area = [int(v) for v in white_stats[label_id]]
            if region_area < total_area * 0.001 or region_area > total_area * 0.25:
                continue
            fill_ratio = region_area / float(max(1, bw * bh))
            if fill_ratio < 0.35 or bw >= w * 0.70 or bh >= h * 0.65:
                continue
            balloon_regions[label_id] = (bx, by, bw, bh, region_area)

        # 3. Step 3: Seeded balloon matching for each text block
        # Group text blocks by enclosing balloon label
        balloon_to_texts: Dict[int, List[DetectedBubble]] = {}
        borderless_texts: List[DetectedBubble] = []

        for tb in raw_text_bubbles:
            tbox = tb.text_box or tb.bounding_box
            tx, ty, tw, th = tbox.x, tbox.y, tbox.width, tbox.height

            # Sample white labels within the text bounding box
            patch = white_labels[ty:min(h, ty + th), tx:min(w, tx + tw)]
            unique_labels, counts = np.unique(patch, return_counts=True)

            best_lbl = None
            best_count = 0
            for lbl, count in zip(unique_labels, counts):
                if lbl in balloon_regions and count > best_count:
                    best_lbl = int(lbl)
                    best_count = count

            if best_lbl is not None:
                if best_lbl not in balloon_to_texts:
                    balloon_to_texts[best_lbl] = []
                balloon_to_texts[best_lbl].append(tb)
            else:
                borderless_texts.append(tb)

        # 4. Step 4: Construct unified DetectedBubble objects
        results: List[DetectedBubble] = []

        # A. Process balloon-enclosed dialogues
        for lbl_id, member_texts in balloon_to_texts.items():
            bx, by, bw, bh, _ = balloon_regions[lbl_id]

            # Merge all member text boxes into one unified text_box
            min_tx = min(t.bounding_box.x for t in member_texts)
            min_ty = min(t.bounding_box.y for t in member_texts)
            max_tx = max(t.bounding_box.right for t in member_texts)
            max_ty = max(t.bounding_box.bottom for t in member_texts)
            unified_text_box = BoundingBox(
                x=min_tx,
                y=min_ty,
                width=max(1, max_tx - min_tx),
                height=max(1, max_ty - min_ty)
            )

            # Ensure balloon bounding_box strictly encloses the text with padding
            pad_x = int(8 * scale)
            pad_y = int(8 * scale)
            fx1 = min(bx, max(0, min_tx - pad_x))
            fy1 = min(by, max(0, min_ty - pad_y))
            fx2 = max(bx + bw, min(w, max_tx + pad_x))
            fy2 = max(by + bh, min(h, max_ty + pad_y))

            # Clip balloon boundary to horizontal panel lines so it never crosses panels
            for py in h_dividers:
                if py > max_ty and py < fy2:
                    fy2 = py - 2
                elif py < min_ty and py > fy1:
                    fy1 = py + 2

            # Clip to vertical panel lines
            for px in v_dividers:
                if px > max_tx and px < fx2:
                    fx2 = px - 2
                elif px < min_tx and px > fx1:
                    fx1 = px + 2

            balloon_box = BoundingBox(
                x=fx1,
                y=fy1,
                width=max(1, fx2 - fx1),
                height=max(1, fy2 - fy1)
            )

            # Average confidence
            avg_conf = sum(t.confidence for t in member_texts) / len(member_texts)
            # Majority direction
            v_count = sum(1 for t in member_texts if t.direction == "vertical")
            direction = "vertical" if v_count >= len(member_texts) - v_count else "horizontal"
            aspect_ratio = round(balloon_box.height / max(1, balloon_box.width), 2)

            bubble = DetectedBubble(
                id=len(results) + 1,
                bounding_box=balloon_box,  # Outer speech bubble
                text_box=unified_text_box,  # Exact inner text envelope
                confidence=round(avg_conf, 2),
                direction=direction,
                aspect_ratio=aspect_ratio,
                detector_type="hybrid"
            )
            results.append(bubble)

        # B. Process borderless / floating dialogues
        for tb in borderless_texts:
            tbox = tb.text_box or tb.bounding_box
            pad = int(10 * scale)
            fx1 = max(0, tbox.x - pad)
            fy1 = max(0, tbox.y - pad)
            fx2 = min(w, tbox.right + pad)
            fy2 = min(h, tbox.bottom + pad)

            balloon_box = BoundingBox(
                x=fx1,
                y=fy1,
                width=max(1, fx2 - fx1),
                height=max(1, fy2 - fy1)
            )

            bubble = DetectedBubble(
                id=len(results) + 1,
                bounding_box=balloon_box,
                text_box=tbox,
                confidence=tb.confidence,
                direction=tb.direction,
                aspect_ratio=round(balloon_box.height / max(1, balloon_box.width), 2),
                detector_type="hybrid"
            )
            results.append(bubble)

        # 5. Step 5: Sort into Manga Reading Order (RTL)
        ordered = sort_manga_reading_order(results, reading_direction="rtl")
        return ordered

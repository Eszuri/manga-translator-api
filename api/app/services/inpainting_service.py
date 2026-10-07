from typing import List, Optional
from PIL import Image
import numpy as np
import cv2

from app.schemas import DetectedBubble


class MangaInpaintingService:

    def __init__(
        self,
        mask_threshold: float = 0.30,
        dilation_kernel_size: int = 5,
        dilation_iterations: int = 2,
        inpaint_radius: int = 3,
        inpaint_method: int = cv2.INPAINT_TELEA
    ):
        self.mask_threshold = mask_threshold
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_iterations = dilation_iterations
        self.inpaint_radius = inpaint_radius
        self.inpaint_method = inpaint_method

    def create_text_mask(
        self,
        image_shape: tuple,
        seg_mask: Optional[np.ndarray] = None,
        bubbles: Optional[List[DetectedBubble]] = None,
        source_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        orig_h, orig_w = image_shape[:2]
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        allowed = np.zeros_like(final_mask) if bubbles is not None else np.full_like(final_mask, 255)
        char_mask = np.zeros_like(final_mask)
        gray = (cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
                if source_image is not None and source_image.ndim == 3 else source_image)

        if seg_mask is not None:
            char_mask = (seg_mask > self.mask_threshold).astype(np.uint8) * 255
            if char_mask.shape != (orig_h, orig_w):
                char_mask = cv2.resize(char_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        if bubbles is None:
            final_mask = char_mask.copy()
        else:
            for b in bubbles:
                box = b.text_box or b.bounding_box
                pad = max(6, round(max(orig_h, orig_w) * 0.006))
                tx1, ty1 = max(0, box.x - pad), max(0, box.y - pad)
                tx2, ty2 = min(orig_w, box.right + pad), min(orig_h, box.bottom + pad)
                if tx2 > tx1 and ty2 > ty1:
                    local_allowed = np.full((ty2 - ty1, tx2 - tx1), 255, np.uint8)
                    if b.bubble_polygon:
                        local_allowed[:] = 0
                        contour = np.array([(x - tx1, y - ty1) for x, y in b.bubble_polygon], np.int32)
                        cv2.fillPoly(local_allowed, [contour], 255)
                        local_allowed = cv2.erode(local_allowed, np.ones((3, 3), np.uint8))
                    allowed[ty1:ty2, tx1:tx2] |= local_allowed
                    patch = char_mask[ty1:ty2, tx1:tx2].copy()
                    if gray is not None:
                        ink = (gray[ty1:ty2, tx1:tx2] < 190).astype(np.uint8)
                        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
                        evidence = cv2.dilate(patch, np.ones((3, 3), np.uint8))
                        has_evidence = bool(np.any(evidence))
                        patch[:] = 0
                        max_dim = max(20, min(box.width, box.height) * 1.8)
                        for label in range(1, count):
                            x, y, w, h, area = stats[label]
                            if (area < 2 or w > max_dim or h > max_dim
                                    or x == 0 or y == 0 or x + w == ink.shape[1] or y + h == ink.shape[0]):
                                continue
                            component = labels == label
                            if has_evidence and not np.any(evidence[component]):
                                continue
                            patch[component] = 255
                    final_mask[ty1:ty2, tx1:tx2] |= patch & local_allowed

                if b.bubble_polygon and gray is not None:
                    polygon = np.asarray(b.bubble_polygon, np.int32)
                    px, py, pw, ph = cv2.boundingRect(polygon)
                    px1, py1 = max(0, px), max(0, py)
                    px2, py2 = min(orig_w, px + pw), min(orig_h, py + ph)
                    if px2 <= px1 or py2 <= py1:
                        continue
                    inner = np.zeros((py2 - py1, px2 - px1), np.uint8)
                    cv2.fillPoly(inner, [polygon - (px1, py1)], 255)
                    inset = max(5, round(min(pw, ph) * 0.045))
                    inner = cv2.erode(inner, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0)
                    if np.count_nonzero(inner) < 400:
                        continue
                    crop_gray = gray[py1:py2, px1:px2]
                    white_fraction = np.mean(crop_gray[inner != 0] > 215)
                    if white_fraction < 0.72:
                        continue
                    extra = ((crop_gray < 185) & (inner != 0)).astype(np.uint8) * 255
                    final_mask[py1:py2, px1:px2] |= extra
                    allowed[py1:py2, px1:px2] |= inner

        if self.dilation_kernel_size > 0 and self.dilation_iterations > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation_kernel_size, self.dilation_kernel_size)
            )
            final_mask = cv2.dilate(final_mask, kernel, iterations=self.dilation_iterations)

        return final_mask & allowed

    def inpaint(
        self,
        image: Image.Image,
        seg_mask: Optional[np.ndarray] = None,
        bubbles: Optional[List[DetectedBubble]] = None,
    ) -> Image.Image:
        if image.width == 0 or image.height == 0:
            return image

        img_rgb = np.array(image.convert("RGB"))
        mask = self.create_text_mask(img_rgb.shape, seg_mask=seg_mask, bubbles=bubbles,
                                     source_image=img_rgb)

        if np.count_nonzero(mask) == 0:
            return image

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(
            img_bgr,
            mask,
            inpaintRadius=self.inpaint_radius,
            flags=self.inpaint_method
        )
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        if bubbles:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            for bubble in bubbles:
                if not bubble.bubble_polygon:
                    continue
                polygon = np.asarray(bubble.bubble_polygon, np.int32)
                interior = np.zeros(gray.shape, np.uint8)
                cv2.fillPoly(interior, [polygon], 255)
                if np.count_nonzero(interior) < 400:
                    continue
                if np.mean(gray[interior != 0] > 215) < 0.87:
                    continue
                interior = cv2.erode(interior, np.ones((3, 3), np.uint8),
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
                inpainted_rgb[interior != 0] = 255
        return Image.fromarray(inpainted_rgb)

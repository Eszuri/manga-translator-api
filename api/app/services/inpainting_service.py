from typing import List, Optional
from PIL import Image
import numpy as np
import cv2

from app.schemas import DetectedBubble


class MangaInpaintingService:
    """
    Service for erasing original Japanese text from manga speech bubbles and panels.
    Combines deep-learning character segmentation masks with morphological dilation
    and fast OpenCV Telea inpainting.
    """

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
        bubbles: Optional[List[DetectedBubble]] = None
    ) -> np.ndarray:
        """
        Builds a binary inpainting mask (255 where text is present, 0 elsewhere).
        """
        orig_h, orig_w = image_shape[:2]
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # 1. Use neural character segmentation mask if available
        if seg_mask is not None:
            char_mask = (seg_mask > self.mask_threshold).astype(np.uint8) * 255
            # Resize if dimensions differ
            if char_mask.shape != (orig_h, orig_w):
                char_mask = cv2.resize(char_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            final_mask = cv2.bitwise_or(final_mask, char_mask)

        # 2. Reinforce mask using detected text_box envelopes
        if bubbles:
            for b in bubbles:
                box = b.text_box or b.bounding_box
                # Safety margin inside text box
                tx1 = max(0, box.x)
                ty1 = max(0, box.y)
                tx2 = min(orig_w, box.right)
                ty2 = min(orig_h, box.bottom)
                if tx2 > tx1 and ty2 > ty1:
                    # If neural mask had no detections in this box, fill the box area
                    box_patch = final_mask[ty1:ty2, tx1:tx2]
                    if np.count_nonzero(box_patch) == 0:
                        final_mask[ty1:ty2, tx1:tx2] = 255

        # 3. Morphological dilation to fully encompass text stroke antialiasing
        if self.dilation_kernel_size > 0 and self.dilation_iterations > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation_kernel_size, self.dilation_kernel_size)
            )
            final_mask = cv2.dilate(final_mask, kernel, iterations=self.dilation_iterations)

        return final_mask

    def inpaint(
        self,
        image: Image.Image,
        seg_mask: Optional[np.ndarray] = None,
        bubbles: Optional[List[DetectedBubble]] = None
    ) -> Image.Image:
        """
        Inpaints the input PIL Image to erase Japanese dialogue text.
        Returns a new PIL Image with clean bubbles.
        """
        if image.width == 0 or image.height == 0:
            return image

        img_rgb = np.array(image.convert("RGB"))
        mask = self.create_text_mask(img_rgb.shape, seg_mask=seg_mask, bubbles=bubbles)

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
        return Image.fromarray(inpainted_rgb)

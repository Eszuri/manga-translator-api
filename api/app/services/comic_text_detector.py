import os
import logging
from time import perf_counter
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort

from app.schemas import BoundingBox, DetectedBubble
from app.core.gpu import configure_gpu_session, verify_gpu_session
from app.core.paths import MODEL_DIR
from app.services.detector import BaseBubbleDetector, sort_manga_reading_order
from app.services.balloon_geometry import refine_text_boxes


DEFAULT_MODEL_PATH = str(MODEL_DIR / "comic-text-detector.onnx")
logger = logging.getLogger(__name__)


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (1024, 1024),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


class ComicTextDetector(BaseBubbleDetector):
    _shared_session: Optional[ort.InferenceSession] = None
    _shared_model_path: Optional[str] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.35,
        num_threads: int = 4
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.num_threads = num_threads

        started = perf_counter()
        logger.info(
            "[startup:detector] Loading comic text detector: %s",
            self.model_path,
            extra={"startup_phase": "loading_detector"},
        )
        self._init_session()

        provider = verify_gpu_session(self.session, "Comic text detector")
        logger.info(
            "[startup:detector] Comic text detector ready in %.1fs (verified GPU provider: %s)",
            perf_counter() - started,
            provider,
        )

    def _init_session(self):
        if (
            ComicTextDetector._shared_session is not None
            and ComicTextDetector._shared_model_path == self.model_path
        ):
            self.session = ComicTextDetector._shared_session
            self.device_name = self._compute_device_name()
            return

        if not os.path.exists(self.model_path):
            try:
                from huggingface_hub import hf_hub_download
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id="mayocream/comic-text-detector-onnx",
                    filename="comic-text-detector.onnx",
                    local_dir=os.path.dirname(self.model_path)
                )
                self.model_path = downloaded
            except Exception as e:
                raise FileNotFoundError(
                    f"Comic Text Detector model not found at '{self.model_path}' "
                    f"and failed to download from Hugging Face: {e}"
                )

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = self.num_threads

        providers = configure_gpu_session(opts)

        self.session = ort.InferenceSession(self.model_path, opts, providers=providers)
        verify_gpu_session(self.session, "Comic text detector")
        self.device_name = self._compute_device_name()
        ComicTextDetector._shared_session = self.session
        ComicTextDetector._shared_model_path = self.model_path

    def _compute_device_name(self) -> str:
        active = self.session.get_providers()
        if "CUDAExecutionProvider" in active:
            return "GPU (NVIDIA CUDA)"
        elif "DmlExecutionProvider" in active:
            return "GPU (DirectML - DirectX Hardware Acceleration)"
        raise RuntimeError(f"Comic text detector has no active GPU provider: {active}")

    def detect_raw(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Tuple[float, float]]:
        img_rgb = np.array(image.convert("RGB"))
        lb_img, r, (dw, dh) = letterbox(img_rgb, new_shape=(1024, 1024))
        inp = (lb_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        blk, seg, det = self.session.run(None, {"images": inp})
        return blk, seg, det, r, (dw, dh)

    def get_unletterboxed_seg(
        self,
        seg: np.ndarray,
        orig_w: int,
        orig_h: int,
        dw: float,
        dh: float
    ) -> np.ndarray:
        mask_h, mask_w = seg.shape[-2:]
        top = int(round(dh - 0.1))
        bottom = mask_h - int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = mask_w - int(round(dw + 0.1))
        cropped = seg[0, 0, top:bottom, left:right]
        resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def detect(self, image: Image.Image, refine: bool = True) -> List[DetectedBubble]:
        orig_w, orig_h = image.size
        if orig_w == 0 or orig_h == 0:
            return []

        blk, seg, _, r, (dw, dh) = self.detect_raw(image)
        self._latest_segmentation = (image, self.get_unletterboxed_seg(seg, orig_w, orig_h, dw, dh))
        preds = blk[0]

        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        obj_conf = preds[:, 4]
        prob_h = preds[:, 5]
        prob_v = preds[:, 6]

        max_cls_prob = np.maximum(prob_h, prob_v)
        scores = obj_conf * max_cls_prob

        valid_mask = scores > self.conf_threshold
        if not np.any(valid_mask):
            return []

        candidate_indices = np.where(valid_mask)[0]
        cx_v = cx[candidate_indices]
        cy_v = cy[candidate_indices]
        w_v = w[candidate_indices]
        h_v = h[candidate_indices]
        scores_v = scores[candidate_indices]
        prob_h_v = prob_h[candidate_indices]
        prob_v_v = prob_v[candidate_indices]

        x1_v = cx_v - w_v / 2.0
        y1_v = cy_v - h_v / 2.0

        nms_boxes = [
            [int(x1_v[i]), int(y1_v[i]), int(w_v[i]), int(h_v[i])]
            for i in range(len(candidate_indices))
        ]
        nms_scores = [float(s) for s in scores_v]

        selected_indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=nms_scores,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold
        )

        if len(selected_indices) == 0:
            return []

        detected_list: List[DetectedBubble] = []

        for item in selected_indices:
            idx = item if isinstance(item, (int, np.integer)) else item[0]

            unpad_x = x1_v[idx] - dw
            unpad_y = y1_v[idx] - dh
            bx = max(0, int(round(unpad_x / r)))
            by = max(0, int(round(unpad_y / r)))
            right = min(orig_w, int(round((unpad_x + w_v[idx]) / r)))
            bottom = min(orig_h, int(round((unpad_y + h_v[idx]) / r)))
            bw, bh = right - bx, bottom - by
            if bw <= 0 or bh <= 0:
                continue

            direction = "vertical" if prob_v_v[idx] >= prob_h_v[idx] else "horizontal"
            aspect_ratio = round(bh / max(1, bw), 2)
            conf = float(scores_v[idx])

            bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

            bubble = DetectedBubble(
                id=len(detected_list) + 1,
                bounding_box=bbox,
                text_box=bbox,
                confidence=round(conf, 2),
                direction=direction,
                aspect_ratio=aspect_ratio
            )
            detected_list.append(bubble)

        if not refine:
            return sort_manga_reading_order(detected_list, reading_direction='rtl')
        segmentation = self._latest_segmentation[1]
        gray = np.array(image.convert('L'))
        refined = []
        for bubble in detected_list:
            for box in refine_text_boxes(bubble.text_box, gray, segmentation, bubble.direction):
                refined.append(bubble.model_copy(update={
                    'bounding_box': box, 'text_box': box,
                    'aspect_ratio': round(box.height / box.width, 2),
                }))
        ordered = sort_manga_reading_order(refined, reading_direction="rtl")
        return ordered

    def get_cached_segmentation(self, image: Image.Image) -> Optional[np.ndarray]:
        cached = getattr(self, '_latest_segmentation', None)
        return cached[1] if cached is not None and cached[0] is image else None

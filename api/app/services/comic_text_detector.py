import os
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort

from app.schemas import BoundingBox, DetectedBubble
from app.services.detector import BaseBubbleDetector, sort_manga_reading_order


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "comic-text-detector.onnx"
)


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (1024, 1024),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resizes image to a 32-pixel multiple while preserving original aspect ratio
    using neutral gray padding (YOLO/Vision standard).
    Returns: (letterboxed_image, scale_ratio, (dw, dh))
    """
    shape = img.shape[:2]  # [height, width]
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
    """
    Deep-learning text detector specifically trained on manga and comics (dmMaze / mayocream architecture).
    Detects text regions, orientation (vertical/horizontal), and character segmentation masks.
    """
    _shared_session: Optional[ort.InferenceSession] = None
    _shared_model_path: Optional[str] = None
    _shared_device: Optional[str] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.35,
        num_threads: int = 4,
        require_gpu: bool = False,
        device: str = "auto"
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.num_threads = num_threads
        
        # Normalize device option
        target_device = "gpu" if require_gpu else device.lower()
        if target_device not in ("auto", "gpu", "cpu"):
            target_device = "auto"
        self.target_device = target_device

        self._init_session()

        if self.target_device == "gpu":
            if not {"CUDAExecutionProvider", "DmlExecutionProvider"}.intersection(self.session.get_providers()):
                raise RuntimeError(
                    "GPU required but neither CUDA nor DirectML is active. "
                    "Use .venv-gpu/Scripts/python.exe and install requirements-gpu.txt."
                )
            # Total lockdown on GPU: do not silently fall back to CPU
            self.session.disable_fallback()

    def _init_session(self):
        """Initializes or reuses the ONNX inference session."""
        if (
            ComicTextDetector._shared_session is not None
            and ComicTextDetector._shared_model_path == self.model_path
            and ComicTextDetector._shared_device == self.target_device
        ):
            self.session = ComicTextDetector._shared_session
            self.device_name = self._compute_device_name()
            return

        # Download from Hugging Face if file doesn't exist locally
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

        available_providers = ort.get_available_providers()
        
        if self.target_device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif self.target_device == "gpu":
            if "CUDAExecutionProvider" in available_providers and hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider"]
            elif "DmlExecutionProvider" in available_providers:
                opts.enable_mem_pattern = False
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                providers = ["DmlExecutionProvider"]
            else:
                raise RuntimeError(
                    "Target device 'gpu' specified, but no GPU provider (CUDA/DirectML) found in ONNX Runtime. "
                    f"Available providers: {available_providers}"
                )
        else:
            # Auto mode: prefer GPU if available, else CPU
            if "CUDAExecutionProvider" in available_providers and hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "DmlExecutionProvider" in available_providers:
                opts.enable_mem_pattern = False
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, opts, providers=providers)
        self.device_name = self._compute_device_name()
        ComicTextDetector._shared_session = self.session
        ComicTextDetector._shared_model_path = self.model_path
        ComicTextDetector._shared_device = self.target_device

    def _compute_device_name(self) -> str:
        active = self.session.get_providers()
        if "CUDAExecutionProvider" in active:
            return "GPU (NVIDIA CUDA)"
        elif "DmlExecutionProvider" in active:
            return "GPU (DirectML - DirectX Hardware Acceleration)"
        return "CPU (Host Processor)"

    def detect_raw(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Tuple[float, float]]:
        """
        Runs model inference with letterbox preprocessing.
        Returns: (blk, seg, det, ratio, (dw, dh))
        """
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
        """
        Unletterboxes segmentation mask [1, 1, 1024, 1024] to original image resolution (orig_h, orig_w).
        """
        top = int(round(dh))
        bottom = int(round(1024 - dh))
        left = int(round(dw))
        right = int(round(1024 - dw))
        cropped = seg[0, 0, top:bottom, left:right]
        resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def detect(self, image: Image.Image) -> List[DetectedBubble]:
        """
        Detects manga text blocks using the Comic Text Detector neural network with Letterbox.
        Returns a list of DetectedBubble objects with text_box populated.
        """
        orig_w, orig_h = image.size
        if orig_w == 0 or orig_h == 0:
            return []

        blk, seg, _, r, (dw, dh) = self.detect_raw(image)
        preds = blk[0]  # Shape: [64512, 7]

        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        obj_conf = preds[:, 4]
        prob_h = preds[:, 5]
        prob_v = preds[:, 6]

        max_cls_prob = np.maximum(prob_h, prob_v)
        scores = obj_conf * max_cls_prob

        # 1. Filter candidates by score threshold
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

        # Convert cx, cy, w, h to x1, y1, w, h in letterbox space for OpenCV NMS
        x1_v = cx_v - w_v / 2.0
        y1_v = cy_v - h_v / 2.0

        nms_boxes = [
            [int(x1_v[i]), int(y1_v[i]), int(w_v[i]), int(h_v[i])]
            for i in range(len(candidate_indices))
        ]
        nms_scores = [float(s) for s in scores_v]

        # 2. Non-Maximum Suppression (NMS)
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

            # Undo letterbox padding and scaling
            unpad_x = x1_v[idx] - dw
            unpad_y = y1_v[idx] - dh
            bx = int(round(unpad_x / r))
            by = int(round(unpad_y / r))
            bw = int(round(w_v[idx] / r))
            bh = int(round(h_v[idx] / r))

            # Clamp coordinates to original image boundaries
            bx = max(0, min(bx, orig_w - 1))
            by = max(0, min(by, orig_h - 1))
            bw = max(1, min(bw, orig_w - bx))
            bh = max(1, min(bh, orig_h - by))

            # Direction: vertical vs horizontal
            direction = "vertical" if prob_v_v[idx] >= prob_h_v[idx] else "horizontal"
            aspect_ratio = round(bh / max(1, bw), 2)
            conf = float(scores_v[idx])

            bbox = BoundingBox(x=bx, y=by, width=bw, height=bh)

            bubble = DetectedBubble(
                id=len(detected_list) + 1,
                bounding_box=bbox,
                text_box=bbox,  # Exact text bounding box from detector
                confidence=round(conf, 2),
                direction=direction,
                aspect_ratio=aspect_ratio,
                detector_type="comic_text_detector"
            )
            detected_list.append(bubble)

        # 3. Sort into Manga Reading Order (RTL by default)
        ordered = sort_manga_reading_order(detected_list, reading_direction="rtl")
        return ordered

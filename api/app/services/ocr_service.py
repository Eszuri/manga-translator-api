import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import re
import logging
from typing import List, Optional
from PIL import Image, ImageDraw
import numpy as np
import onnxruntime as ort
import jaconv
from transformers import ViTImageProcessorPil as ViTImageProcessor, BertJapaneseTokenizer

from app.schemas import DetectedBubble

logger = logging.getLogger(__name__)

DEFAULT_OCR_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "manga-ocr")


class MangaOcrService:
    """
    Japanese Manga OCR Engine based on Vision Transformer (ViT + RoBERTa) in ONNX format.
    Optimized for high-accuracy recognition of vertical/horizontal text, kanji, and furigana.
    Supports GPU DirectML acceleration and CPU execution.
    """
    _shared_instance: Optional["MangaOcrService"] = None
    _shared_device: Optional[str] = None

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "auto",
        require_gpu: bool = False,
        num_threads: int = 4
    ):
        self.model_dir = model_dir or DEFAULT_OCR_DIR
        self.num_threads = num_threads

        target_device = "gpu" if require_gpu else device.lower()
        if target_device not in ("auto", "gpu", "cpu"):
            target_device = "auto"
        self.target_device = target_device

        self._init_models()

    def _init_models(self):
        """Initializes tokenizer, processor, and ONNX Runtime sessions."""
        encoder_path = os.path.join(self.model_dir, "encoder_model.onnx")
        decoder_path = os.path.join(self.model_dir, "decoder_model.onnx")

        model_source = self.model_dir if os.path.exists(encoder_path) else "mayocream/manga-ocr-onnx"

        logger.info(f"Loading OCR processor and tokenizer from: {model_source}")
        self.processor = ViTImageProcessor.from_pretrained(model_source)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_source)

        self.eos_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id

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
                    "Target device 'gpu' requested for OCR, but no GPU provider (CUDA/DirectML) found. "
                    f"Available providers: {available_providers}"
                )
        else:
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

        logger.info(f"Manga-OCR active providers: {providers}")
        self.encoder_session = ort.InferenceSession(encoder_path, opts, providers=providers)
        self.decoder_session = ort.InferenceSession(decoder_path, opts, providers=providers)

        if self.target_device == "gpu":
            self.encoder_session.disable_fallback()
            self.decoder_session.disable_fallback()

        self.device_name = self._compute_device_name()

    def _compute_device_name(self) -> str:
        active = self.encoder_session.get_providers()
        if "CUDAExecutionProvider" in active:
            return "GPU (NVIDIA CUDA)"
        elif "DmlExecutionProvider" in active:
            return "GPU (DirectML - DirectX Hardware Acceleration)"
        return "CPU (Host Processor)"

    def recognize_crop(self, image: Image.Image, max_length: int = 300) -> str:
        """
        Runs ViT encoder and autoregressive greedy decoder to extract Japanese text from a crop.
        """
        img_rgb = image.convert("L").convert("RGB")
        pixel_values = self.processor(img_rgb, return_tensors="np").pixel_values

        encoder_outputs = self.encoder_session.run(None, {"pixel_values": pixel_values})
        last_hidden_state = encoder_outputs[0]

        input_ids = np.array([[self.bos_token_id]], dtype=np.int64)

        for _ in range(max_length):
            decoder_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": last_hidden_state
            }
            try:
                logits = self.decoder_session.run(None, decoder_inputs)[0]
            except Exception as e:
                logger.error(f"Decoder run failed: {e}")
                break

            next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            input_ids = np.concatenate([input_ids, np.array([[next_token]], dtype=np.int64)], axis=-1)

            if next_token == self.eos_token_id:
                break

        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        text = "".join(text.split())
        text = text.replace("…", "...")
        text = re.sub(r"[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
        text = jaconv.h2z(text, ascii=True, digit=True)
        return text

    def recognize_bubble(
        self,
        full_image: Image.Image,
        bubble: DetectedBubble,
        padding: int = 6
    ) -> str:
        """
        Crops text area with safety padding from full image and performs OCR.
        """
        target = bubble.text_box if bubble.text_box is not None else bubble.bounding_box
        w, h = full_image.size

        x1 = max(0, target.x - padding)
        y1 = max(0, target.y - padding)
        x2 = min(w, target.right + padding)
        y2 = min(h, target.bottom + padding)

        if x2 <= x1 or y2 <= y1:
            return ""

        crop = full_image.crop((x1, y1, x2, y2))
        if bubble.bubble_polygon:
            # Exclude neighbouring art/balloons from padded OCR crops.
            shape = Image.new('L', crop.size, 0)
            ImageDraw.Draw(shape).polygon(
                [(x - x1, y - y1) for x, y in bubble.bubble_polygon], fill=255)
            crop = Image.composite(crop.convert('RGB'), Image.new('RGB', crop.size, 'white'), shape)
        return self.recognize_crop(crop)

    def recognize_all_bubbles(
        self,
        full_image: Image.Image,
        bubbles: List[DetectedBubble],
        padding: int = 6
    ) -> List[DetectedBubble]:
        """
        Extracts OCR text for all bubbles in the list.
        Populates bubble.text for each DetectedBubble.
        """
        for bubble in bubbles:
            bubble.text = self.recognize_bubble(full_image, bubble, padding=padding)
        return bubbles


def get_ocr_service(device: str = "auto", require_gpu: bool = False) -> MangaOcrService:
    """Returns singleton instance of MangaOcrService."""
    target_dev = "gpu" if require_gpu else device.lower()
    if (
        MangaOcrService._shared_instance is not None
        and MangaOcrService._shared_device == target_dev
    ):
        return MangaOcrService._shared_instance

    instance = MangaOcrService(device=device, require_gpu=require_gpu)
    MangaOcrService._shared_instance = instance
    MangaOcrService._shared_device = target_dev
    return instance

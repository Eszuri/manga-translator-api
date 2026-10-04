"""Resolve model files independently of the source checkout or frozen bundle."""
import os
from pathlib import Path


MODEL_DIR = Path(os.getenv("MANGA_MODEL_DIR") or Path(__file__).resolve().parents[1] / "models")

REQUIRED_MODEL_FILES = (
    "comic-text-detector.onnx",
    "manga-ocr/encoder_model.onnx",
    "manga-ocr/decoder_model.onnx",
    "manga-ocr/config.json",
    "manga-ocr/preprocessor_config.json",
    "manga-ocr/tokenizer_config.json",
    "manga-ocr/vocab.txt",
)


def missing_model_files(directory: Path) -> list[str]:
    return [name for name in REQUIRED_MODEL_FILES
            if not (directory / name).is_file() or (directory / name).stat().st_size == 0]

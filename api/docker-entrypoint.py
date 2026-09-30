"""Validate mounted assets before starting the container API."""
import os
from pathlib import Path
import sys

import uvicorn

from app.core.gpu import required_gpu_provider


if __name__ == "__main__":
    model_dir = Path("/app/app/models")
    required_files = [
        "comic-text-detector.onnx",
        "manga-ocr/encoder_model.onnx",
        "manga-ocr/decoder_model.onnx",
        "manga-ocr/config.json",
        "manga-ocr/preprocessor_config.json",
        "manga-ocr/tokenizer_config.json",
        "manga-ocr/vocab.txt",
    ]
    missing = [name for name in required_files if not (model_dir / name).is_file()]
    if missing:
        print("Missing model files in the mounted api/app/models directory:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)

    try:
        provider = required_gpu_provider()
    except RuntimeError as exc:
        print(f"Strict GPU startup rejected: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Strict GPU inference enabled with {provider}")

    reload_enabled = os.getenv("API_RELOAD", "False").lower() in ("true", "1", "yes")

    # A single process avoids duplicating ONNX models and their memory usage.
    # Development can enable reload while production keeps an immutable process.
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=1,
        reload=reload_enabled,
        reload_dirs=["/app/app"] if reload_enabled else None,
    )

"""Validate mounted assets before starting the container API."""
from pathlib import Path
import sys

import uvicorn


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

    # A single process avoids duplicating ONNX models and their memory usage.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)

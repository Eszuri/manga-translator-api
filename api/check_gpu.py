"""Verify GPU executes model operators and benchmark one local manga page."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time

import onnxruntime as ort
from PIL import Image

from app.services.comic_text_detector import ComicTextDetector, DEFAULT_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(__file__).parent / "image test/018.jpg")
    args = parser.parse_args()
    print(f"ONNX Runtime: {ort.__version__}", flush=True)
    print(f"Available providers: {ort.get_available_providers()}", flush=True)
    available = ort.get_available_providers()
    provider = next((p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available), None)
    if provider is None:
        raise SystemExit("GPU is unavailable. Run with .venv-gpu/Scripts/python.exe.")
    if not Path(DEFAULT_MODEL_PATH).is_file():
        raise SystemExit(f"Model missing: {DEFAULT_MODEL_PATH}")

    if provider == "CUDAExecutionProvider":
        ort.preload_dlls()
    with tempfile.TemporaryDirectory(prefix="manga-gpu-check-") as temp_dir:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        if provider == "DmlExecutionProvider":
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(temp_dir) / "profile")
        session = ort.InferenceSession(
            DEFAULT_MODEL_PATH, options,
            providers=[provider, "CPUExecutionProvider"],
        )
        print(f"Active providers: {session.get_providers()}", flush=True)
        if provider not in session.get_providers():
            raise SystemExit("GPU failed to initialize; refusing to benchmark CPU fallback.")
        session.disable_fallback()
        ComicTextDetector._shared_session = session
        ComicTextDetector._shared_model_path = DEFAULT_MODEL_PATH
        detector = ComicTextDetector(require_gpu=True)
        with Image.open(args.image) as image:
            image.load()
            for label in ("First inference", "Warm inference"):
                started = time.perf_counter()
                bubbles = detector.detect(image)
                print(f"{label}: {time.perf_counter() - started:.3f}s, {len(bubbles)} text blocks", flush=True)
        profile_path = session.end_profiling()
        events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        counts = Counter(
            event.get("args", {}).get("provider") for event in events
            if event.get("cat") == "Node" and event.get("args", {}).get("provider")
        )
        print(f"Executed operator events: {dict(counts)}", flush=True)
        if not counts[provider]:
            raise SystemExit("No GPU operator execution was recorded.")
        print(f"PASS: model operators executed via {provider}.", flush=True)
        ComicTextDetector._shared_session = None
        del detector, session


if __name__ == "__main__":
    main()

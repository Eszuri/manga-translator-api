import os
from pathlib import Path


MODEL_DIR = Path(os.getenv("MANGA_MODEL_DIR") or Path(__file__).resolve().parents[3] / "models")

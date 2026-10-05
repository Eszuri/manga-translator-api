"""Console entry point for the standalone Windows API package."""

import argparse
import logging
import os
from pathlib import Path
import sys

if "__compiled__" not in globals() and not getattr(sys, "frozen", False):
    api_dir = Path(__file__).resolve().parents[1]
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))


def package_dir() -> Path:
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def positive_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Port must be an integer from 1 to 65535.") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be an integer from 1 to 65535.")
    return port


def positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Timeout must be greater than zero.") from exc
    if not 0 < timeout < float("inf"):
        raise argparse.ArgumentTypeError("Timeout must be greater than zero.")
    return timeout


def resolve_external_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Manga Translator API in a console without the desktop GUI."
    )
    parser.add_argument("--host", help="Bind address; use 0.0.0.0 for trusted network access.")
    parser.add_argument("--port", type=positive_port, help="HTTP port (default: 8000).")
    parser.add_argument("--model-dir", help="Model directory (default: models beside the executable).")
    parser.add_argument("--env-file", help="Environment file (default: .env beside the executable).")
    parser.add_argument("--translator", choices=("google", "llm"), help="Default translator.")
    parser.add_argument("--llm-base-url", help="LLM API base URL; API key stays in .env.")
    parser.add_argument("--llm-model", help="LLM model ID.")
    parser.add_argument("--llm-timeout", type=positive_timeout, help="LLM timeout in seconds.")
    parser.add_argument("--check", action="store_true", help="Check configuration, OCR assets, and GPU provider without starting HTTP.")
    args = parser.parse_args(argv)

    base = package_dir()
    env_file = resolve_external_path(args.env_file, base) if args.env_file else (
        base / (".env" if "__compiled__" in globals() or getattr(sys, "frozen", False) else "api/.env")
    )
    if args.env_file and not env_file.is_file():
        parser.error(f"Environment file does not exist: {env_file}")

    # All settings must be established before importing backend modules.
    from dotenv import load_dotenv

    if env_file.is_file():
        load_dotenv(env_file, override=False)
    os.environ["API_LOAD_DOTENV"] = "False"
    os.environ["API_RELOAD"] = "False"
    os.environ["API_DEBUG"] = "False"
    os.environ["GPU_PROVIDER"] = "DmlExecutionProvider"

    overrides = {
        "API_HOST": args.host,
        "API_PORT": str(args.port) if args.port is not None else None,
        "DEFAULT_TRANSLATOR": args.translator,
        "LLM_BASE_URL": args.llm_base_url,
        "LLM_MODEL": args.llm_model,
        "LLM_TIMEOUT_SECONDS": str(args.llm_timeout) if args.llm_timeout is not None else None,
    }
    for name, value in overrides.items():
        if value is not None:
            os.environ[name] = value

    configured_models = args.model_dir or os.getenv("MANGA_MODEL_DIR")
    model_dir = resolve_external_path(configured_models, base) if configured_models else base / "models"
    if not model_dir.is_dir() and not configured_models and base.parent.name.casefold() == "dist":
        candidate = base.parent.parent / "models"
        if candidate.is_dir():
            model_dir = candidate
    os.environ["MANGA_MODEL_DIR"] = str(model_dir)

    try:
        from app.core.config import settings
    except ValueError as exc:
        parser.error(f"Invalid configuration: {exc}")

    from app.core.paths import missing_model_files

    missing = missing_model_files(model_dir)
    if missing:
        parser.error("Missing or empty model files in " + str(model_dir) + ": " + ", ".join(missing))

    from app.core.gpu import required_gpu_provider

    provider = required_gpu_provider()
    from app.main import app

    if args.check:
        # Load packaged OCR assets without creating multi-gigabyte GPU sessions.
        import ssl
        import certifi
        from app.services.ocr_service import ViTImageProcessor, BertJapaneseTokenizer

        ViTImageProcessor.from_pretrained(str(model_dir / "manga-ocr"), local_files_only=True)
        BertJapaneseTokenizer.from_pretrained(str(model_dir / "manga-ocr"), local_files_only=True)
        ssl.create_default_context(cafile=certifi.where())

        print(f"Models: {model_dir}\nGPU provider: {provider}\nConfiguration check passed.")
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    print(f"Starting Manga Translator API at http://{settings.HOST}:{settings.PORT}", flush=True)
    print(f"Models: {model_dir} | GPU provider: {provider}", flush=True)
    server = uvicorn.Server(uvicorn.Config(app, host=settings.HOST, port=settings.PORT, workers=1, reload=False))
    server.run()
    return 0 if server.started else 1


if __name__ == "__main__":
    raise SystemExit(main())

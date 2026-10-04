"""Run the API in a child process owned by the desktop application."""
import json
import copy
import errno
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
import sys
import threading
import traceback

from server_gui.paths import icon_path, normalize_model_dir, user_data_dir
from server_gui.preferences import load_preferences


class _RedactingFormatter(logging.Formatter):
    def __init__(self, secret: str, fmt="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt=None):
        super().__init__(fmt, datefmt=datefmt)
        self.secret = secret

    def format(self, record):
        value = super().format(record)
        return value.replace(self.secret, "[REDACTED]") if self.secret else value


def _error_summary(record):
    if record.exc_info and record.exc_info[1] is not None:
        return f"{record.exc_info[0].__name__}: {record.exc_info[1]}"
    message = record.getMessage().strip()
    if "Traceback (most recent call last):" in message:
        return next((line.strip() for line in reversed(message.splitlines()) if line.strip()), message)
    return message


class _GuiFormatter(_RedactingFormatter):
    def __init__(self, secret: str):
        super().__init__(secret, "%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S")

    def format(self, record):
        concise = copy.copy(record)
        concise.msg = _error_summary(record)
        concise.args = ()
        concise.exc_info = concise.exc_text = None
        return super().format(concise)


class _GuiLogHandler(logging.StreamHandler):
    """Keep actionable messages in the window and complete diagnostics on disk."""

    def __init__(self, secret: str):
        super().__init__(sys.stdout)
        self.startup_error = ""
        self.setFormatter(_GuiFormatter(secret))

    def emit(self, record):
        phases = {
            "loading_detector": "Loading text detector on GPU",
            "loading_ocr_processor": "Loading OCR processor and tokenizer",
            "loading_ocr_encoder": "Loading OCR encoder on GPU",
            "loading_ocr_decoder": "Loading OCR decoder on GPU",
            "models_ready": "GPU models ready; starting HTTP server",
        }
        phase = getattr(record, "startup_phase", None)
        if phase in phases:
            print("GUI_EVENT " + json.dumps({"phase": phase, "message": phases[phase]}), flush=True)
        message = record.getMessage()
        if record.name == "uvicorn.access":
            # Normal browser traffic and health polling add no useful GUI status.
            try:
                if int(record.args[-1]) < 400 or "/internal/gui/health " in message:
                    return
            except (TypeError, ValueError, IndexError, KeyError):
                pass
        if record.levelno >= logging.ERROR:
            summary = _error_summary(record)
            if summary and summary != "Application startup failed. Exiting.":
                self.startup_error = summary
        super().emit(record)


def configure_logging(secret: str) -> _GuiLogHandler:
    directory = user_data_dir() / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    gui = _GuiLogHandler(secret)
    disk = RotatingFileHandler(directory / "server.log", maxBytes=2 * 1024 * 1024,
                               backupCount=3, encoding="utf-8")
    disk.setFormatter(_RedactingFormatter(secret))
    disk.addFilter(lambda record: record.name != "uvicorn.access" or "/internal/gui/health " not in record.getMessage())
    logging.basicConfig(level=logging.INFO, handlers=[gui, disk], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return gui


def _command_lines():
    """Read the private pipe without holding CRT locks during native imports."""
    if os.name != "nt" or not hasattr(sys.stdin, "fileno"):
        yield from sys.stdin
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt

    # A blocking Python/CRT read can deadlock NumPy DLL initialization on Windows.
    # ReadFile waits only on the inherited OS pipe and leaves CRT streams unlocked.
    try:
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
    except (OSError, ValueError):
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    read = kernel.ReadFile
    read.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                     ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    read.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(256)
    count = wintypes.DWORD()
    pending = b""
    while read(handle, buffer, len(buffer), ctypes.byref(count), None) and count.value:
        pending += buffer.raw[:count.value]
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")


def run_worker(settings_path: Path) -> int:
    prefs = load_preferences(settings_path)
    gui_log = configure_logging(prefs.api_key)
    model_dir = normalize_model_dir(prefs.model_dir)
    host = "0.0.0.0" if prefs.allow_network else "127.0.0.1"
    stop_requested = threading.Event()
    server = None

    def read_commands():
        try:
            for line in _command_lines():
                if line.strip() == "STOP":
                    break
        finally:
            # Observe parent exit during imports/model setup as well as while serving.
            stop_requested.set()
            if server is not None:
                server.should_exit = True

    threading.Thread(target=read_commands, name="desktop-control", daemon=True).start()
    # Set every backend option before importing modules with import-time settings.
    overrides = {
        "API_LOAD_DOTENV": "False", "API_HOST": host,
        "API_PORT": str(prefs.port), "API_DEBUG": "False", "API_RELOAD": "False",
        "GPU_PROVIDER": "DmlExecutionProvider", "MANGA_MODEL_DIR": str(model_dir),
        "LLM_API_KEY": prefs.api_key, "LLM_BASE_URL": prefs.effective_llm_base_url(),
        "LLM_MODEL": prefs.llm_model, "LLM_TIMEOUT_SECONDS": str(prefs.llm_timeout),
        "DEFAULT_TRANSLATOR": prefs.translator,
        "ALLOWED_ORIGINS": "http://localhost:3000,http://127.0.0.1:3000",
        "CORS_ALLOW_CREDENTIALS": "False", "MAX_UPLOAD_BYTES": "10485760",
        "MAX_IMAGE_PIXELS": "40000000", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    os.environ.update(overrides)
    try:
        from app.core import paths as model_paths
        # Discovery may have imported this module before the worker environment existed.
        model_paths.MODEL_DIR = model_dir
        missing = model_paths.missing_model_files(model_dir)
        if missing:
            raise FileNotFoundError("Missing or empty model files: " + ", ".join(missing))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
            if os.name == "nt":
                check.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                check.bind((host, prefs.port))
            except OSError as exc:
                if getattr(exc, "winerror", None) == 10048 or exc.errno == errno.EADDRINUSE:
                    raise RuntimeError(f"Port {prefs.port} is already in use. Stop the server using it, or select another API port in Settings.") from exc
                raise
        if stop_requested.is_set():
            return 0
        from app.core.gpu import required_gpu_provider
        required_gpu_provider()
        if stop_requested.is_set():
            return 0
        print("GUI_EVENT " + json.dumps({"phase": "loading_models"}), flush=True)
        from app.main import app
        from fastapi import HTTPException, Request
        from fastapi.responses import FileResponse
        import secrets
        import uvicorn

        instance = os.environ["MANGA_GUI_INSTANCE"]

        @app.get("/internal/gui/health", include_in_schema=False)
        async def desktop_health(request: Request):
            if not secrets.compare_digest(request.headers.get("X-Server-Instance", ""), instance):
                raise HTTPException(status_code=403, detail="This endpoint belongs to the desktop controller.")
            return {"instance": instance, "gpu_provider": app.state.gpu_provider}

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            path = icon_path()
            if not path.is_file():
                raise HTTPException(status_code=404, detail="The application icon is unavailable.")
            return FileResponse(path, media_type="image/png")

        server = uvicorn.Server(uvicorn.Config(
            app, host=host, port=prefs.port, workers=1, reload=False,
            log_config=None, timeout_graceful_shutdown=10))

        if stop_requested.is_set():
            return 0
        server.run()
        if not server.started and not stop_requested.is_set():
            message = gui_log.startup_error or "Server startup failed. Open the log folder for details."
            if prefs.api_key:
                message = message.replace(prefs.api_key, "[REDACTED]")
            print("GUI_EVENT " + json.dumps({"error": message}), flush=True)
        return 0 if server.started else 1
    except Exception as exc:
        message = str(exc).replace(prefs.api_key, "[REDACTED]") if prefs.api_key else str(exc)
        print("GUI_EVENT " + json.dumps({"error": message}), flush=True)
        trace = traceback.format_exc()
        if prefs.api_key:
            trace = trace.replace(prefs.api_key, "[REDACTED]")
        logging.error("Server startup failed:\n%s", trace)
        return 1

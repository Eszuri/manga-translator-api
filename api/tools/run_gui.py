"""Desktop entry point; the same executable can run its owned server worker."""
import argparse
import io
import json
import os
from pathlib import Path
import sys


def restore_worker_streams():
    """Windowed frozen apps have None streams even with inherited QProcess pipes."""
    if os.name == "nt":
        import ctypes
        import msvcrt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.argtypes = [ctypes.c_ulong]
        kernel.GetStdHandle.restype = ctypes.c_void_p
        for name, code, mode in (("stdin", -10, "rb"), ("stdout", -11, "wb"), ("stderr", -12, "wb")):
            if getattr(sys, name) is None:
                handle = kernel.GetStdHandle(code & 0xFFFFFFFF)
                if handle and handle != ctypes.c_void_p(-1).value:
                    flags = (os.O_RDONLY if name == "stdin" else os.O_WRONLY) | os.O_BINARY
                    descriptor = msvcrt.open_osfhandle(handle, flags)
                    stream = io.TextIOWrapper(os.fdopen(descriptor, mode, buffering=0), encoding="utf-8", line_buffering=True)
                    setattr(sys, name, stream)
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    if sys.stdin is None:
        sys.stdin = io.StringIO("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manga Translator desktop server")
    parser.add_argument("--server-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--settings-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--check-gui", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_gui:
        restore_worker_streams()
    if args.server_worker:
        restore_worker_streams()
        try:
            from server_gui.worker import run_worker
            if args.settings_file is None:
                raise ValueError("A settings file is required for the internal server worker.")
            return run_worker(args.settings_file)
        except Exception as exc:
            print("GUI_EVENT " + json.dumps({"error": f"Worker configuration failed: {type(exc).__name__}: {exc}"}), flush=True)
            return 1
    from PySide6.QtCore import QLockFile
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication, QMessageBox
    from server_gui.paths import user_data_dir
    from server_gui.window import ServerWindow

    app = QApplication(sys.argv[:1])
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("Manga Translator Server")
    if args.check_gui:
        if args.settings_file is None:
            raise ValueError("The GUI build check requires an isolated settings path.")
        window = ServerWindow(settings_path=args.settings_file, auto_start=False)
        window.show()
        app.processEvents()
        window.close()
        print("GUI_CHECK_OK", flush=True)
        return 0
    directory = user_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(directory / "desktop.lock"))
    if not lock.tryLock(0):
        QMessageBox.information(None, "Already open", "Manga Translator Server is already open. Use the existing window.")
        return 1
    window = ServerWindow(settings_path=args.settings_file, auto_start=False)
    window.show()
    try:
        return app.exec()
    finally:
        lock.unlock()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    raise SystemExit(main())

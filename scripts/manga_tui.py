
from __future__ import annotations

import argparse
import asyncio
import atexit
import ctypes
import ipaddress
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import warnings
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    from ctypes import wintypes

    try:
        from asyncio.base_subprocess import BaseSubprocessTransport
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _orig_subprocess_del = BaseSubprocessTransport.__del__

        def _safe_subprocess_del(self, _warn=warnings.warn):
            try:
                _orig_subprocess_del(self, _warn)
            except Exception:
                pass

        BaseSubprocessTransport.__del__ = _safe_subprocess_del

        _orig_pipe_del = _ProactorBasePipeTransport.__del__

        def _safe_pipe_del(self, _warn=warnings.warn):
            try:
                _orig_pipe_del(self, _warn)
            except Exception:
                pass

        _ProactorBasePipeTransport.__del__ = _safe_pipe_del
    except Exception:
        pass

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Footer, RichLog

for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
API_DIR = PROJECT_ROOT / "api"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_ORIGINAL = API_DIR / "Images" / "original Images"
IMAGES_BUILD = API_DIR / "Images" / "build Images"
EXTENSION_DIR = PROJECT_ROOT / "extension"
DIST_DIR = PROJECT_ROOT / "dist"
ENV_FILE = API_DIR / ".env"

PYTHON_EXE = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ACTIVE_CHILD_PIDS: set[int] = set()
EXTENSION_BUILD_DONE = threading.Event()
EXTENSION_BUILD_DONE.set()
EXTENSION_BUILD_RUNNING = False
EXTENSION_BUILD_LOCK = threading.Lock()
_CTRL_HANDLER_REF = None


def cleanup_pid(pid: int) -> None:
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=5)
    except Exception:
        pass


def _close_proc_transport(proc: asyncio.subprocess.Process | None) -> None:
    if proc is None:
        return
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


def cleanup_all_processes(wait_extension: bool = True) -> None:
    global EXTENSION_BUILD_RUNNING
    if wait_extension and EXTENSION_BUILD_RUNNING:
        try:
            EXTENSION_BUILD_DONE.wait(timeout=5.0)
        except Exception:
            pass

    pids = list(ACTIVE_CHILD_PIDS)
    for pid in pids:
        cleanup_pid(pid)
    ACTIVE_CHILD_PIDS.clear()


def win32_ctrl_handler(_ctrl_type: int) -> bool:
    cleanup_all_processes(wait_extension=True)
    return False


def setup_process_lifecycle() -> None:
    global _CTRL_HANDLER_REF
    atexit.register(cleanup_all_processes)

    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            PHANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            _CTRL_HANDLER_REF = PHANDLER_ROUTINE(win32_ctrl_handler)
            kernel32.SetConsoleCtrlHandler(_CTRL_HANDLER_REF, True)
        except Exception:
            pass

    def _sig_handler(_signum, _frame):
        cleanup_all_processes(wait_extension=True)
        sys.exit(0)

    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            try:
                signal.signal(getattr(signal, sig), _sig_handler)
            except Exception:
                pass


def get_server_port() -> str:
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("API_PORT="):
                        return line.strip().split("=", 1)[1].strip().strip("\"'")
        except Exception:
            pass
    return "8000"


def get_network_access_urls(port: str) -> dict[str, list[str]]:
    urls: dict[str, list[str]] = {
        "local": [f"http://127.0.0.1:{port}"],
        "lan": [],
        "tailscale": [],
    }
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None):
            if item[0] == socket.AF_INET:
                ip = item[4][0]
                if ip.startswith("127."):
                    continue
                ip_obj = ipaddress.IPv4Address(ip)
                url = f"http://{ip}:{port}"
                if ip_obj in ipaddress.IPv4Network("100.64.0.0/10"):
                    if url not in urls["tailscale"]:
                        urls["tailscale"].append(url)
                elif ip_obj.is_private:
                    if url not in urls["lan"]:
                        urls["lan"].append(url)
    except Exception:
        pass
    return urls


def find_chromium_browser() -> str | None:
    candidates = [
        os.environ.get("CHROME_PATH"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return cand
    for name in ("chrome", "google-chrome", "chromium", "msedge", "brave"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build_extension_package_sync(log_func=None) -> bool:
    global EXTENSION_BUILD_RUNNING
    if log_func is None:
        from rich import print as rprint
        log_func = rprint
    if not EXTENSION_BUILD_LOCK.acquire(blocking=False):
        log_func("[yellow]An extension build is already running.[/yellow]")
        return False
    EXTENSION_BUILD_DONE.clear()
    EXTENSION_BUILD_RUNNING = True
    try:
        manifest = EXTENSION_DIR / "manifest.json"
        if not manifest.exists():
            log_func(f"[red]Manifest ekstensi tidak ditemukan: {manifest}[/red]")
            return False

        DIST_DIR.mkdir(parents=True, exist_ok=True)
        zip_output = DIST_DIR / "manga-translator.zip"
        crx_output = DIST_DIR / "manga-translator.crx"
        key_output = DIST_DIR / "manga-translator.pem"

        log_func("[cyan]Mengemas extension menjadi manga-translator.zip...[/cyan]")
        exclude_parts = {".git", "__pycache__"}
        exclude_files = {".DS_Store", "Thumbs.db"}

        try:
            with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in EXTENSION_DIR.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(EXTENSION_DIR)
                        if any(p in rel.parts for p in exclude_parts):
                            continue
                        if file_path.name in exclude_files or file_path.suffix == ".pyc":
                            continue
                        zf.write(file_path, str(rel).replace("\\", "/"))

            log_func(f"[green]✔ Berhasil membuat: {zip_output}[/green]")
        except Exception as e:
            log_func(f"[red]Gagal membuat file ZIP: {e}[/red]")
            return False

        browser = find_chromium_browser()
        if not browser:
            log_func("[yellow]Browser Chromium tidak ditemukan. Hanya paket ZIP yang dibuat.[/yellow]")
            return True

        log_func(f"[cyan]Mengemas file CRX menggunakan: {Path(browser).name}...[/cyan]")
        with tempfile.TemporaryDirectory() as temp_dir:
            staging_dir = Path(temp_dir) / "extension"
            try:
                shutil.copytree(EXTENSION_DIR, staging_dir)
                args = [browser, f"--pack-extension={staging_dir}", "--no-message-box"]
                if key_output.exists():
                    args.append(f"--pack-extension-key={key_output}")

                chrome_proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ACTIVE_CHILD_PIDS.add(chrome_proc.pid)
                try:
                    chrome_proc.wait(timeout=30)
                finally:
                    if chrome_proc.poll() is None:
                        cleanup_pid(chrome_proc.pid)
                        chrome_proc.wait(timeout=5)
                    ACTIVE_CHILD_PIDS.discard(chrome_proc.pid)

                gen_crx = Path(temp_dir) / "extension.crx"
                gen_pem = Path(temp_dir) / "extension.pem"

                if gen_crx.exists():
                    shutil.move(str(gen_crx), str(crx_output))
                    if gen_pem.exists() and not key_output.exists():
                        shutil.move(str(gen_pem), str(key_output))
                    log_func(f"[green]✔ Berhasil membuat: {crx_output}[/green]")
                else:
                    log_func("[yellow]CRX tidak dihasilkan browser (Gunakan file ZIP).[/yellow]")
            except Exception as e:
                log_func(f"[yellow]Peringatan pembuatan CRX: {e} (Paket ZIP tetap siap).[/yellow]")

        return True
    finally:
        EXTENSION_BUILD_RUNNING = False
        EXTENSION_BUILD_DONE.set()
        EXTENSION_BUILD_LOCK.release()


def format_log_line(prefix: str, text: str) -> str:
    if "[Translate]" in text:
        idx = text.find("[Translate]")
        payload = text[idx + len("[Translate]"):].strip()
        if payload.startswith("Fallback"):
            tag = "[bold yellow][Translate][/bold yellow]"
        elif payload.startswith("Done"):
            tag = "[bold green][Translate][/bold green]"
        else:
            tag = "[bold cyan][Translate][/bold cyan]"
        return f"{prefix} {tag} {payload}"
    return f"{prefix} {text}"


class MangaTranslatorTUI(App):

    TITLE = "Manga Translator"

    CSS = """
    Screen {
        layout: vertical;
        padding: 0 1;
    }

    Button {
        border: none !important;
        height: 3;
    }

    Button:focus {
        text-style: none !important;
        background-tint: transparent !important;
    }

    #action-bar {
        height: auto;
        margin: 1 0;
    }

    #action-bar Button {
        border: none !important;
        height: 3;
        margin-right: 1;
    }

    RichLog {
        height: 1fr;
        background: #0d1117;
        color: #e6edf3;
        border: solid $primary-darken-1;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("1", "toggle_server", "Server"),
        ("2", "trigger_image_build", "Images"),
        ("3", "trigger_ext_build", "Extension"),
        ("4", "open_docs", "Docs"),
        ("5", "open_build_folder", "Output"),
        ("c", "clear_log", "Clear"),
        ("d", "toggle_dark", "Dark/Light"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.port = get_server_port()
        self.server_proc: asyncio.subprocess.Process | None = None
        self.worker_proc: asyncio.subprocess.Process | None = None
        self.current_task: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="action-bar"):
            for btn in (
                Button("Start Server", id="btn-server", variant="success"),
                Button("Process Images", id="btn-images", variant="primary"),
                Button("Build Extension", id="btn-ext"),
                Button("Docs", id="btn-docs"),
                Button("Output", id="btn-folder"),
                Button("Clear", id="btn-clear"),
            ):
                btn.can_focus = False
                yield btn
        yield RichLog(id="terminal-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.update_status()
        self.query_one("#terminal-log", RichLog).focus()
        self.set_interval(2.0, self.update_status)
        if not VENV_PYTHON.exists():
            self.log_msg("[red]⚠ .venv missing. Silakan siapkan virtual environment terlebih dahulu.[/red]")
        has_detector = (MODELS_DIR / "comic-text-detector.onnx").exists()
        has_ocr = (MODELS_DIR / "manga-ocr").exists()
        if not (has_detector and has_ocr):
            self.log_msg("[yellow]⚠ Model detector atau OCR belum lengkap di folder models/.[/yellow]")

    def log_msg(self, text: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        log = self.query_one("#terminal-log", RichLog)
        log.write(f"[dim]{now}[/dim] {text}")

    def update_status(self) -> None:
        btn_server = self.query_one("#btn-server", Button)
        btn_images = self.query_one("#btn-images", Button)

        is_server_alive = self.server_proc is not None and self.server_proc.returncode is None
        is_worker_alive = self.worker_proc is not None and self.worker_proc.returncode is None

        if is_server_alive:
            btn_server.label = "Stop Server"
            btn_server.variant = "error"
        else:
            btn_server.label = "Start Server"
            btn_server.variant = "success"

        if is_worker_alive:
            btn_images.label = "Stop"
            btn_images.variant = "error"
        else:
            btn_images.label = "Process Images"
            btn_images.variant = "primary"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-server":
            self.action_toggle_server()
        elif bid == "btn-images":
            self.action_trigger_image_build()
        elif bid == "btn-ext":
            self.action_trigger_ext_build()
        elif bid == "btn-docs":
            self.action_open_docs()
        elif bid == "btn-folder":
            self.action_open_build_folder()
        elif bid == "btn-clear":
            self.action_clear_log()

    def action_toggle_server(self) -> None:
        if self.server_proc and self.server_proc.returncode is None:
            self.stop_server()
        else:
            if self.worker_proc and self.worker_proc.returncode is None:
                self.stop_worker()
            self.start_server()

    @work(exclusive=True, group="server_group")
    async def start_server(self) -> None:
        net_urls = get_network_access_urls(self.port)
        self.log_msg(f"[green]Starting Server (:{self.port})...[/green]")
        self.log_msg(f"[dim]  ● Local: {net_urls['local'][0]}[/dim]")
        for lan in net_urls["lan"]:
            self.log_msg(f"[cyan]  ● LAN: {lan}[/cyan]")
        for ts in net_urls["tailscale"]:
            self.log_msg(f"[magenta]  ● Tailscale: {ts}[/magenta]")

        try:
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            cmd = [str(PYTHON_EXE), "-m", "tools.run_local"]
            self.server_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(API_DIR),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            ACTIVE_CHILD_PIDS.add(self.server_proc.pid)
            self.update_status()

            proc = self.server_proc
            while proc and proc.returncode is None:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                self.query_one("#terminal-log", RichLog).write(format_log_line("[dim][SRV][/dim]", text))

            if proc and proc.returncode is None:
                try:
                    await proc.wait()
                except Exception:
                    pass
            self.log_msg("[yellow]Server stopped[/yellow]")
        except asyncio.CancelledError:
            proc = self.server_proc
            if proc and proc.returncode is None:
                cleanup_pid(proc.pid)
            raise
        except Exception as e:
            self.log_msg(f"[red]Server error: {e}[/red]")
        finally:
            if self.server_proc:
                ACTIVE_CHILD_PIDS.discard(self.server_proc.pid)
                _close_proc_transport(self.server_proc)
            self.server_proc = None
            self.update_status()

    def stop_server(self) -> None:
        if self.server_proc and self.server_proc.returncode is None:
            proc = self.server_proc
            self.server_proc = None
            pid = proc.pid
            ACTIVE_CHILD_PIDS.discard(pid)
            self.log_msg(f"[yellow]Stopping server (PID: {pid})...[/yellow]")
            cleanup_pid(pid)
            _close_proc_transport(proc)
            self.update_status()

    def action_trigger_image_build(self) -> None:
        if (self.current_task == "Building Extension"
                or (self.current_task is not None and self.worker_proc is None)):
            self.log_msg("[yellow]Wait for the current task to finish.[/yellow]")
            return
        if self.worker_proc and self.worker_proc.returncode is None:
            self.stop_worker()
        else:
            if self.server_proc and self.server_proc.returncode is None:
                self.stop_server()
            self.current_task = "Processing Images"
            self.run_image_pipeline()

    def stop_worker(self) -> None:
        if self.worker_proc and self.worker_proc.returncode is None:
            proc = self.worker_proc
            self.worker_proc = None
            self.current_task = None
            pid = proc.pid
            ACTIVE_CHILD_PIDS.discard(pid)
            self.log_msg(f"[yellow]Stopping task (PID: {pid})...[/yellow]")
            cleanup_pid(pid)
            _close_proc_transport(proc)
            self.update_status()

    @work(exclusive=True, group="task_group")
    async def run_image_pipeline(self) -> None:
        if self.worker_proc and self.worker_proc.returncode is None:
            return

        try:
            if not IMAGES_ORIGINAL.is_dir():
                raise FileNotFoundError(f"Folder not found: {IMAGES_ORIGINAL}")
            has_images = any(f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                             for f in IMAGES_ORIGINAL.iterdir())
            if not has_images:
                raise FileNotFoundError(f"No images found in: {IMAGES_ORIGINAL}")
        except OSError as exc:
            self.log_msg(f"[red]Image error: {exc}[/red]")
            self.current_task = None
            self.update_status()
            return

        cmd = [str(PYTHON_EXE), "-m", "tools.build_local"]
        self.log_msg("[cyan]Processing images...[/cyan]")
        self.current_task = "Processing Images"
        self.update_status()

        try:
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            self.worker_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(API_DIR),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            ACTIVE_CHILD_PIDS.add(self.worker_proc.pid)

            proc = self.worker_proc
            while proc and proc.returncode is None:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                self.query_one("#terminal-log", RichLog).write(format_log_line("[cyan][IMG][/cyan]", text))

            if proc and proc.returncode is None:
                try:
                    await proc.wait()
                except Exception:
                    pass

            if proc and proc.returncode == 0:
                self.log_msg("[green]Images finished[/green]")
            elif proc and proc.returncode is not None:
                self.log_msg(f"[red]Images failed (code: {proc.returncode})[/red]")
        except asyncio.CancelledError:
            proc = self.worker_proc
            if proc and proc.returncode is None:
                cleanup_pid(proc.pid)
            raise
        except Exception as e:
            self.log_msg(f"[red]Image error: {e}[/red]")
        finally:
            if self.worker_proc:
                ACTIVE_CHILD_PIDS.discard(self.worker_proc.pid)
                _close_proc_transport(self.worker_proc)
            self.worker_proc = None
            self.current_task = None
            self.update_status()

    def action_trigger_ext_build(self) -> None:
        if (self.current_task is not None
                or (self.worker_proc is not None and self.worker_proc.returncode is None)):
            self.log_msg("[yellow]Wait for the current task to finish.[/yellow]")
            return
        self.current_task = "Building Extension"
        self.run_extension_build()

    @work(exclusive=True, group="extension_group")
    async def run_extension_build(self) -> None:
        if self.worker_proc and self.worker_proc.returncode is None:
            return

        self.log_msg("[cyan]Building extension...[/cyan]")
        self.current_task = "Building Extension"
        self.update_status()
        try:
            loop = asyncio.get_running_loop()
            def log_from_thread(message):
                loop.call_soon_threadsafe(self.log_msg, message)

            success = await loop.run_in_executor(None, build_extension_package_sync, log_from_thread)
            if success:
                self.log_msg("[green]Extension built[/green]")
            else:
                self.log_msg("[red]Extension build failed[/red]")
        except Exception as e:
            self.log_msg(f"[red]Extension error: {e}[/red]")
        finally:
            self.worker_proc = None
            self.current_task = None
            self.update_status()

    def action_open_docs(self) -> None:
        url = f"http://127.0.0.1:{self.port}/docs"
        webbrowser.open(url)
        self.log_msg(f"Docs: {url}")

    def action_open_build_folder(self) -> None:
        target = IMAGES_BUILD if IMAGES_BUILD.exists() else PROJECT_ROOT
        os.startfile(str(target))
        self.log_msg(f"Folder: {target.name}")

    def action_clear_log(self) -> None:
        self.query_one("#terminal-log", RichLog).clear()

    def action_quit(self) -> None:
        _close_proc_transport(self.worker_proc)
        _close_proc_transport(self.server_proc)
        if self.worker_proc and self.worker_proc.returncode is None:
            self.stop_worker()
        if self.server_proc and self.server_proc.returncode is None:
            self.stop_server()
        cleanup_all_processes(wait_extension=True)
        self.exit()

    def on_unmount(self) -> None:
        _close_proc_transport(self.worker_proc)
        _close_proc_transport(self.server_proc)
        if self.worker_proc and self.worker_proc.returncode is None:
            self.stop_worker()
        if self.server_proc and self.server_proc.returncode is None:
            self.stop_server()
        cleanup_all_processes(wait_extension=True)


def run_cli_server():
    print("Menjalankan API Server lokal...")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen([str(PYTHON_EXE), "-m", "tools.run_local"], cwd=str(API_DIR), env=env)
    ACTIVE_CHILD_PIDS.add(proc.pid)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if proc.poll() is None:
            cleanup_pid(proc.pid)
        ACTIVE_CHILD_PIDS.discard(proc.pid)


def run_cli_images():
    if not IMAGES_ORIGINAL.exists():
        print(f"[FAILED] Folder gambar tidak ditemukan: {IMAGES_ORIGINAL}")
        sys.exit(1)
    has_images = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in IMAGES_ORIGINAL.iterdir())
    if not has_images:
        print(f"[FAILED] Tidak ada gambar JPEG, PNG, atau WebP di: {IMAGES_ORIGINAL}")
        sys.exit(1)

    cmd = [str(PYTHON_EXE), "-m", "tools.build_local"]
    print(f"Memproses gambar: {' '.join(cmd[1:])}")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(cmd, cwd=str(API_DIR), env=env)
    ACTIVE_CHILD_PIDS.add(proc.pid)
    try:
        ret = proc.wait()
    except KeyboardInterrupt:
        ret = 1
    finally:
        if proc.poll() is None:
            cleanup_pid(proc.pid)
        ACTIVE_CHILD_PIDS.discard(proc.pid)
    return ret


def main():
    setup_process_lifecycle()
    parser = argparse.ArgumentParser(description="Manga Translator Runner & TUI", add_help=False)
    parser.add_argument("--server", action="store_true", help="Jalankan FastAPI Server lokal langsung")
    parser.add_argument("--process-images", action="store_true", help="Jalankan build gambar lokal langsung")
    parser.add_argument("--build-ext", action="store_true", help="Jalankan build browser extension langsung")
    parser.add_argument("-h", "--help", action="store_true", help="Tampilkan bantuan")

    args, _ = parser.parse_known_args()

    if args.help:
        print("Penggunaan:")
        print("  scripts\\run-tui.bat                     : Buka antarmuka TUI interaktif (default)")
        print("  scripts\\run-tui.bat --server            : Jalankan API server langsung")
        print("  scripts\\run-tui.bat --process-images    : Jalankan build gambar lokal langsung")
        print("  scripts\\run-tui.bat --build-ext         : Build browser extension langsung")
        return

    if args.server:
        return run_cli_server()

    if args.process_images:
        return run_cli_images()

    if args.build_ext:
        return 0 if build_extension_package_sync() else 1

    app = MangaTranslatorTUI()
    app.run()


if __name__ == "__main__":
    raise SystemExit(main())

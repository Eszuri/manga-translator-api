"""Manga Translator Control Center & Unified Runner
Menggabungkan seluruh runner (Server API, Build Gambar, Build Extension)
dalam satu skrip Python native dan menyediakan antarmuka TUI interaktif.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

# Pastikan encoding console Windows mendukung karakter Unicode/Rich
for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Direktori proyek
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


def get_server_port() -> str:
    """Membaca port API dari api/.env."""
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("API_PORT="):
                        return line.strip().split("=", 1)[1].strip().strip("\"'")
        except Exception:
            pass
    return "8000"


def find_chromium_browser() -> str | None:
    """Mencari binary browser Chromium untuk build CRX."""
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
    """Memaketkan folder extension/ menjadi .zip dan .crx secara native."""
    if log_func is None:
        from rich import print as rprint
        log_func = rprint

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

    # Kemas CRX jika Chromium tersedia
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

            proc = subprocess.run(args, capture_output=True, timeout=60)
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


class MangaTranslatorTUI(App):
    """TUI satu layar interaktif untuk Manga Translator."""

    TITLE = "Manga Translator Control Center"

    CSS = """
    Screen {
        layout: vertical;
        padding: 0 1;
    }

    #status-bar {
        height: 3;
        background: $surface;
        border: round $primary;
        padding: 0 1;
        content-align: center middle;
        margin-bottom: 1;
    }

    #action-bar {
        height: auto;
        margin-bottom: 1;
    }

    #action-bar Button {
        margin-right: 1;
    }

    #extra-args {
        margin-bottom: 1;
    }

    RichLog {
        height: 1fr;
        background: #0d1117;
        color: #e6edf3;
        border: solid $accent;
        padding: 1;
    }
    """

    BINDINGS = [
        ("1", "toggle_server", "Start/Stop Server"),
        ("2", "trigger_image_build", "Process Images"),
        ("3", "trigger_ext_build", "Build Extension"),
        ("4", "open_docs", "Swagger Docs"),
        ("5", "open_build_folder", "Buka Folder Hasil"),
        ("c", "clear_log", "Clear Log"),
        ("d", "toggle_dark", "Dark/Light"),
        ("q", "quit", "Keluar"),
    ]

    def __init__(self):
        super().__init__()
        self.port = get_server_port()
        self.server_proc: asyncio.subprocess.Process | None = None
        self.worker_proc: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="status-bar")

        with Horizontal(id="action-bar"):
            yield Button("▶ [1] Start Server", id="btn-server", variant="success")
            yield Button("⚡ [2] Process Images", id="btn-images", variant="primary")
            yield Button("📦 [3] Build Extension", id="btn-ext", variant="warning")
            yield Button("🌐 [4] Docs", id="btn-docs")
            yield Button("📁 [5] Folder Hasil", id="btn-folder")
            yield Button("🧹 [C] Clear Log", id="btn-clear")

        yield Input(
            placeholder="Opsi tambahan build gambar (opsional, contoh: --limit 1 atau --image 001.jpg)...",
            id="extra-args",
        )

        yield RichLog(id="terminal-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.update_status_bar()
        self.log_msg(f"[green]✔ Manga Translator TUI Siap.[/green] Lingkungan: [cyan]{PYTHON_EXE.name}[/cyan]")
        self.log_msg("[dim]Tekan tombol angka 1 - 5, C untuk bersihkan log, atau Q untuk keluar.[/dim]")
        self.set_interval(3.0, self.update_status_bar)

    def log_msg(self, text: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        log = self.query_one("#terminal-log", RichLog)
        log.write(f"[dim]{now}[/dim] {text}")

    # ==========================
    # STATUS BAR
    # ==========================

    def update_status_bar(self) -> None:
        status_bar = self.query_one("#status-bar", Static)
        btn_server = self.query_one("#btn-server", Button)

        # Status Server
        if self.server_proc and self.server_proc.returncode is None:
            server_badge = f"[green]● SERVER ACTIVE (:{self.port})[/green]"
            btn_server.label = "⏹ [1] Stop Server"
            btn_server.variant = "error"
        else:
            server_badge = "[red]○ SERVER STOPPED[/red]"
            btn_server.label = "▶ [1] Start Server"
            btn_server.variant = "success"

        # Status Venv / GPU
        venv_badge = "[green]Venv: .venv Siap[/green]" if VENV_PYTHON.exists() else "[yellow]Venv: Default[/yellow]"

        # Status Model
        has_detector = (MODELS_DIR / "comic-text-detector.onnx").exists()
        has_ocr = (MODELS_DIR / "manga-ocr").exists()
        models_badge = "[green]Models: OK[/green]" if (has_detector and has_ocr) else "[red]Models: Kurang[/red]"

        # Hitung Gambar
        img_count = 0
        if IMAGES_ORIGINAL.exists():
            img_count = len([f for f in IMAGES_ORIGINAL.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
        images_badge = f"Images: [cyan]{img_count} file[/cyan]"

        status_bar.update(f"{server_badge}   │   {venv_badge}   │   {models_badge}   │   {images_badge}")

    # ==========================
    # EVENT HANDLERS
    # ==========================

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

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_trigger_image_build()

    # ==========================
    # ACTIONS
    # ==========================

    def action_toggle_server(self) -> None:
        if self.server_proc and self.server_proc.returncode is None:
            self.stop_server()
        else:
            self.start_server()

    @work(exclusive=True)
    async def start_server(self) -> None:
        if self.server_proc and self.server_proc.returncode is None:
            return

        self.log_msg(f"[green]>>> Memulai API Server pada port {self.port}...[/green]")
        try:
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            cmd = [str(PYTHON_EXE), "-m", "tools.run_local"]
            self.server_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(API_DIR),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self.update_status_bar()

            while self.server_proc and self.server_proc.returncode is None:
                line = await self.server_proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                self.query_one("#terminal-log", RichLog).write(f"[dim][SRV][/dim] {text}")

            await self.server_proc.wait()
            self.log_msg("[red]>>> API Server dihentikan.[/red]")
        except Exception as e:
            self.log_msg(f"[red]Error server: {e}[/red]")
        finally:
            self.server_proc = None
            self.update_status_bar()

    def stop_server(self) -> None:
        if self.server_proc and self.server_proc.returncode is None:
            pid = self.server_proc.pid
            self.log_msg(f"[yellow]Menghentikan server (PID: {pid})...[/yellow]")
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            except Exception:
                try:
                    self.server_proc.terminate()
                except Exception:
                    pass
            self.server_proc = None
            self.update_status_bar()

    def action_trigger_image_build(self) -> None:
        self.run_image_pipeline()

    @work(exclusive=True)
    async def run_image_pipeline(self) -> None:
        if self.worker_proc and self.worker_proc.returncode is None:
            self.log_msg("[yellow]Tugas lain sedang berjalan. Harap tunggu hingga selesai.[/yellow]")
            return

        # Validasi folder gambar asli
        if not IMAGES_ORIGINAL.exists():
            self.log_msg(f"[red][FAILED] Folder gambar sumber tidak ditemukan: {IMAGES_ORIGINAL}[/red]")
            return

        has_images = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in IMAGES_ORIGINAL.iterdir())
        if not has_images:
            self.log_msg(f"[red][FAILED] Tidak ada gambar JPEG, PNG, atau WebP di: {IMAGES_ORIGINAL}[/red]")
            return

        args_input = self.query_one("#extra-args", Input).value.strip()
        cmd = [str(PYTHON_EXE), "-m", "tools.build_local", "--device", "gpu"]

        if args_input:
            try:
                extra = shlex.split(args_input)
                cmd.extend(extra)
            except Exception:
                cmd.extend(args_input.split())

        self.log_msg(f"[cyan]>>> Memproses Gambar: {' '.join(cmd[1:])}[/cyan]")

        try:
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            self.worker_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(API_DIR),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            while self.worker_proc and self.worker_proc.returncode is None:
                line = await self.worker_proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                self.query_one("#terminal-log", RichLog).write(f"[cyan][IMG][/cyan] {text}")

            await self.worker_proc.wait()
            if self.worker_proc.returncode == 0:
                self.log_msg("[green]✔ Pemrosesan gambar selesai! Hasil di api/Images/build Images[/green]")
            else:
                self.log_msg(f"[red]Pemrosesan gambar gagal (exit code: {self.worker_proc.returncode})[/red]")
        except Exception as e:
            self.log_msg(f"[red]Error saat memproses gambar: {e}[/red]")
        finally:
            self.worker_proc = None

    def action_trigger_ext_build(self) -> None:
        self.run_extension_build()

    @work(exclusive=True)
    async def run_extension_build(self) -> None:
        if self.worker_proc and self.worker_proc.returncode is None:
            self.log_msg("[yellow]Tugas lain sedang berjalan. Harap tunggu.[/yellow]")
            return

        self.log_msg("[cyan]>>> Memulai Build Browser Extension...[/cyan]")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, build_extension_package_sync, self.log_msg)
        except Exception as e:
            self.log_msg(f"[red]Error saat build extension: {e}[/red]")
        finally:
            self.worker_proc = None

    def action_open_docs(self) -> None:
        url = f"http://127.0.0.1:{self.port}/docs"
        webbrowser.open(url)
        self.log_msg(f"Membuka browser docs: {url}")

    def action_open_build_folder(self) -> None:
        target = IMAGES_BUILD if IMAGES_BUILD.exists() else PROJECT_ROOT
        os.startfile(str(target))
        self.log_msg(f"Membuka folder hasil: {target}")

    def action_clear_log(self) -> None:
        self.query_one("#terminal-log", RichLog).clear()

    def on_unmount(self) -> None:
        for proc in (self.server_proc, self.worker_proc):
            if proc and proc.returncode is None:
                try:
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                except Exception:
                    pass


# ==========================================
# CLI ENTRY POINT (CLI Headless & TUI Mode)
# ==========================================

def run_cli_server():
    """Menjalankan server langsung dari terminal."""
    print("Menjalankan API Server lokal...")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([str(PYTHON_EXE), "-m", "tools.run_local"], cwd=str(API_DIR), env=env)


def run_cli_images(extra_args: list[str]):
    """Menjalankan build gambar langsung dari terminal."""
    if not IMAGES_ORIGINAL.exists():
        print(f"[FAILED] Folder gambar tidak ditemukan: {IMAGES_ORIGINAL}")
        sys.exit(1)
    has_images = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in IMAGES_ORIGINAL.iterdir())
    if not has_images:
        print(f"[FAILED] Tidak ada gambar JPEG, PNG, atau WebP di: {IMAGES_ORIGINAL}")
        sys.exit(1)

    cmd = [str(PYTHON_EXE), "-m", "tools.build_local", "--device", "gpu"] + extra_args
    print(f"Memproses gambar: {' '.join(cmd[1:])}")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(cmd, cwd=str(API_DIR), env=env)
    sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Manga Translator Runner & TUI", add_help=False)
    parser.add_argument("--server", action="store_true", help="Jalankan FastAPI Server lokal langsung")
    parser.add_argument("--process-images", action="store_true", help="Jalankan build gambar lokal langsung")
    parser.add_argument("--build-ext", action="store_true", help="Jalankan build browser extension langsung")
    parser.add_argument("-h", "--help", action="store_true", help="Tampilkan bantuan")

    args, remaining = parser.parse_known_args()

    if args.help:
        print("Penggunaan:")
        print("  scripts\\run-tui.bat                     : Buka antarmuka TUI interaktif (default)")
        print("  scripts\\run-tui.bat --server            : Jalankan API server langsung")
        print("  scripts\\run-tui.bat --process-images    : Jalankan build gambar langsung (terima argumen)")
        print("  scripts\\run-tui.bat --build-ext         : Build browser extension langsung")
        return

    if args.server:
        run_cli_server()
        return

    if args.process_images:
        run_cli_images(remaining)
        return

    if args.build_ext:
        build_extension_package_sync()
        return

    # Default: Buka TUI Interaktif
    app = MangaTranslatorTUI()
    app.run()


if __name__ == "__main__":
    main()

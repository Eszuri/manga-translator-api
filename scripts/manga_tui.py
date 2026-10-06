"""Manga Translator Control Center (Simplified TUI)
TUI minimalis, cepat, dan bersih untuk menjalankan server, memproses gambar,
serta mem-build extension browser.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

# Direktori proyek
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
API_DIR = PROJECT_ROOT / "api"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_ORIGINAL = API_DIR / "Images" / "original Images"
IMAGES_BUILD = API_DIR / "Images" / "build Images"
DIST_DIR = PROJECT_ROOT / "dist"
ENV_FILE = API_DIR / ".env"

PYTHON_EXE = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


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


class MangaTranslatorTUI(App):
    """TUI minimalis satu layar untuk Manga Translator."""

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

        # 1. Status Bar Ringkas
        yield Static(id="status-bar")

        # 2. Tombol Aksi Utama
        with Horizontal(id="action-bar"):
            yield Button("▶ [1] Start Server", id="btn-server", variant="success")
            yield Button("⚡ [2] Process Images", id="btn-images", variant="primary")
            yield Button("📦 [3] Build Extension", id="btn-ext", variant="warning")
            yield Button("🌐 [4] Docs", id="btn-docs")
            yield Button("📁 [5] Folder Hasil", id="btn-folder")
            yield Button("🧹 [C] Clear Log", id="btn-clear")

        # 3. Input Opsi Tambahan (Opsional)
        yield Input(
            placeholder="Opsi tambahan build gambar (opsional, contoh: --limit 1 atau --image 001.jpg)...",
            id="extra-args",
        )

        # 4. Terminal Output Log Penuh
        yield RichLog(id="terminal-log", wrap=True, highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        self.update_status_bar()
        self.log_msg(f"[green]✔ Manga Translator TUI Siap.[/green] Menggunakan: [cyan]{PYTHON_EXE.name}[/cyan]")
        self.log_msg("[dim]Gunakan tombol atau hotkey angka 1 - 5 / C / Q.[/dim]")
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

        # Server State
        if self.server_proc and self.server_proc.returncode is None:
            server_badge = f"[green]● SERVER ACTIVE (:{self.port})[/green]"
            btn_server.label = "⏹ [1] Stop Server"
            btn_server.variant = "error"
        else:
            server_badge = "[red]○ SERVER STOPPED[/red]"
            btn_server.label = "▶ [1] Start Server"
            btn_server.variant = "success"

        # GPU / Venv
        venv_badge = "[green]GPU: Siap[/green]" if VENV_PYTHON.exists() else "[yellow]GPU: Standar[/yellow]"

        # Models
        has_detector = (MODELS_DIR / "comic-text-detector.onnx").exists()
        has_ocr = (MODELS_DIR / "manga-ocr").exists()
        models_badge = "[green]Models: OK[/green]" if (has_detector and has_ocr) else "[red]Models: Kurang[/red]"

        # Images Count
        img_count = 0
        if IMAGES_ORIGINAL.exists():
            img_count = len([f for f in IMAGES_ORIGINAL.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
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
        # Menekan Enter di input langsung menjalankan build gambar dengan argumen tersebut
        self.action_trigger_image_build()

    # ==========================
    # ACTIONS
    # ==========================

    def action_toggle_server(self) -> None:
        """Nyalakan atau hentikan server."""
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
            cmd = [str(PYTHON_EXE), "-m", "tools.run_local"]
            self.server_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(API_DIR),
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
            self.log_msg("[yellow]Tugas lain sedang berjalan. Hentikan dulu atau tunggu selesai.[/yellow]")
            return

        # Ambil opsi tambahan dari input field
        args_input = self.query_one("#extra-args", Input).value.strip()
        cmd = [str(PYTHON_EXE), "-m", "tools.build_local", "--device", "gpu"]

        if args_input:
            try:
                extra = shlex.split(args_input)
                cmd.extend(extra)
            except Exception:
                cmd.extend(args_input.split())

        self.log_msg(f"[cyan]>>> Memulai Build Gambar: {' '.join(cmd[1:])}[/cyan]")

        try:
            self.worker_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(API_DIR),
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
                self.log_msg("[green]✔ Build gambar selesai! Hasil di folder api/Images/build Images[/green]")
            else:
                self.log_msg(f"[red]Build gambar selesai dengan kode: {self.worker_proc.returncode}[/red]")
        except Exception as e:
            self.log_msg(f"[red]Error build gambar: {e}[/red]")
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
            cmd = ["cmd", "/c", "scripts\\build-extension.bat < NUL"]
            self.worker_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            while self.worker_proc and self.worker_proc.returncode is None:
                line = await self.worker_proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                self.query_one("#terminal-log", RichLog).write(f"[yellow][EXT][/yellow] {text}")

            await self.worker_proc.wait()
            if self.worker_proc.returncode == 0:
                self.log_msg("[green]✔ Extension siap di folder dist/[/green]")
            else:
                self.log_msg(f"[red]Build extension gagal (kode: {self.worker_proc.returncode})[/red]")
        except Exception as e:
            self.log_msg(f"[red]Error build extension: {e}[/red]")
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
        """Membersihkan subproses ketika TUI ditutup."""
        for proc in (self.server_proc, self.worker_proc):
            if proc and proc.returncode is None:
                try:
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                except Exception:
                    pass


if __name__ == "__main__":
    app = MangaTranslatorTUI()
    app.run()

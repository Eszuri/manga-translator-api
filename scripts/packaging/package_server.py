"""Build a portable Windows server without including models or local secrets."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "dist" / "MangaTranslatorServer"
WORK = ROOT / "build" / "server-gui"


def running_package_processes(package: Path) -> list[int]:
    """Find only processes launched from this package's executable path."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    enumerate_processes = kernel.K32EnumProcesses
    enumerate_processes.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
                                   ctypes.POINTER(wintypes.DWORD)]
    enumerate_processes.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    capacity = 1024
    while True:
        process_ids = (wintypes.DWORD * capacity)()
        used = wintypes.DWORD()
        if not enumerate_processes(process_ids, ctypes.sizeof(process_ids), ctypes.byref(used)):
            raise ctypes.WinError(ctypes.get_last_error())
        if used.value < ctypes.sizeof(process_ids):
            break
        capacity *= 2
    executable = os.path.normcase(str((package / "MangaTranslatorServer.exe").resolve()))
    matches = []
    for process_id in process_ids[:used.value // ctypes.sizeof(wintypes.DWORD)]:
        # Querying the path does not require opening the process for termination.
        handle = kernel.OpenProcess(0x1000, False, process_id)
        if not handle:
            continue  # System/protected processes and processes that already exited.
        try:
            size = wintypes.DWORD(32768)
            image = ctypes.create_unicode_buffer(size.value)
            if kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
                if os.path.normcase(str(Path(image.value).resolve())) == executable:
                    matches.append(process_id)
        finally:
            kernel.CloseHandle(handle)
    return sorted(matches)


def require_closed_package() -> None:
    processes = running_package_processes(OUTPUT)
    if processes:
        ids = ", ".join(map(str, processes))
        raise RuntimeError(
            f"Cannot update {OUTPUT}: its GUI or server worker is still running (PID {ids}). "
            "Close Manga Translator Server, wait for its server to stop, and run "
            "package-server-gui.bat again. No running process was stopped."
        )


def existing_manifest() -> dict[str, int]:
    path = OUTPUT / "package-manifest.json"
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read the existing package manifest at {path}: {exc}. "
                           "The existing package was not updated.") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"The existing package manifest at {path} must be a JSON object.")
    for name, size in manifest.items():
        relative = Path(name)
        if (not name or relative.is_absolute() or relative.drive or relative.root
                or ".." in relative.parts or type(size) is not int or size < 0):
            raise RuntimeError(f"The existing package manifest has an invalid entry: {name!r}. "
                               "The existing package was not updated.")
    return manifest


def output_runtime() -> Path:
    output = OUTPUT.resolve()
    runtime = (OUTPUT / "_internal").resolve()
    if runtime == output or not runtime.is_relative_to(output):
        raise RuntimeError("The package's _internal folder points outside its output directory. "
                           "The existing package was not updated.")
    return runtime


def build_environment() -> dict[str, str]:
    environment = {name.upper(): value for name, value in os.environ.items()}
    windows = Path(environment["SYSTEMROOT"])
    environment["PATH"] = os.pathsep.join(map(str, (
        Path(sys.executable).parent, Path(sys.base_prefix), windows / "System32", windows,
    )))
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def check_packaged_gui(package: Path, staging: Path) -> None:
    environment = build_environment()
    environment.update(QT_QPA_PLATFORM="offscreen", LOCALAPPDATA=str(staging / "gui-check-data"))
    try:
        check = subprocess.run([
            str(package / "MangaTranslatorServer.exe"), "--check-gui",
            "--settings-file", str(staging / "gui-check-settings.json"),
        ], cwd=package, env=environment, capture_output=True, text=True, encoding="utf-8",
           errors="replace", timeout=90)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("The packaged GUI did not finish its startup check within 90 seconds. "
                           "The existing package was not updated.") from exc
    if check.returncode != 0 or "GUI_CHECK_OK" not in check.stdout:
        raise RuntimeError(f"Packaged GUI check failed ({check.returncode}):\n{check.stdout}\n{check.stderr}")
    print("Packaged Qt GUI startup check passed.", flush=True)


def retire_obsolete_runtime(manifest: dict[str, int]) -> None:
    """Move obsolete owned runtime files aside; never touch models or settings."""
    previous = existing_manifest()
    stale = set(previous) - set(manifest)
    # These known incompatible files must also be removed from older untracked packages.
    stale.update(str(Path("_internal") / name) for name in ("icuuc.dll", "icuin.dll", "icudt.dll"))
    backup = None
    runtime = output_runtime()
    sources = []
    for name in sorted(stale):
        # Only runtime files recorded by the package can be retired.
        if not Path(name).parts or Path(name).parts[0] != "_internal":
            continue
        source = (OUTPUT / name).resolve()
        if not source.is_relative_to(runtime) or not source.is_relative_to(OUTPUT.resolve()):
            raise RuntimeError(f"Runtime file points outside the package: {name}. "
                               "The existing package was not updated.")
        if not source.is_file():
            continue
        sources.append(source)
    # Validate all targets before moving any files.
    for source in sources:
        if backup is None:
            backup = Path(tempfile.mkdtemp(prefix="retired-runtime-", dir=WORK)).resolve()
            if not backup.is_relative_to(WORK.resolve()):
                raise RuntimeError("Runtime backup path is outside the build directory.")
        destination = backup / source.relative_to(runtime)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    if backup:
        print(f"Obsolete runtime files moved to: {backup}", flush=True)


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("Build this Windows DirectML package on Windows.")
    require_closed_package()
    existing_manifest()
    output_runtime()
    import onnxruntime as ort
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("The Windows package requires onnxruntime-directml with DmlExecutionProvider.")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    WORK.mkdir(parents=True, exist_ok=True)
    # A fresh staging directory prevents PyInstaller from deleting existing external models.
    with tempfile.TemporaryDirectory(prefix="staging-", dir=WORK) as staging:
        staging_path = Path(staging)
        subprocess.run([sys.executable, "-m", "PyInstaller", "--distpath", str(staging_path),
                        "--workpath", str(WORK / "cache"),
                        str(ROOT / "scripts/packaging/server_gui.spec")], cwd=ROOT,
                       env=build_environment(), check=True)
        package = staging_path / "MangaTranslatorServer"
        files = sorted(path for path in package.rglob("*") if path.is_file())
        forbidden = [path.relative_to(package) for path in files
                     if path.suffix.lower() in (".onnx", ".pt", ".pth", ".safetensors")
                     or path.name in (".env", "settings.json", "server.log")
                     or path.name.lower() in ("icuuc.dll", "icuin.dll", "icudt.dll")]
        if forbidden:
            raise RuntimeError(f"Package contains excluded files: {forbidden}")
        required = ("DirectML.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll")
        names = {path.name for path in files}
        for name in required:
            if name not in names:
                raise RuntimeError(f"Required GPU runtime was not packaged: {name}")
        manifest = {str(path.relative_to(package)): path.stat().st_size for path in files}
        # Test the actual windowed executable before updating the published package.
        check_packaged_gui(package, staging_path)
        require_closed_package()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        retire_obsolete_runtime(manifest)
        shutil.copytree(package, OUTPUT, dirs_exist_ok=True)
        # Never copy, delete, or overwrite the user's model directory.
        (OUTPUT / "models").mkdir(exist_ok=True)
        (OUTPUT / "package-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (OUTPUT / "START-HERE.txt").write_text(
            "Manga Translator Server (Windows x64)\n\n"
            "Run MangaTranslatorServer.exe. Python is bundled.\n"
            "The server stays stopped on launch. Click Start server when ready.\n"
            "Models are external. Put comic-text-detector.onnx and the manga-ocr folder\n"
            "inside models, or choose an existing model folder in Settings.\n"
            "Keep the executable together with its _internal folder.\n"
            "Google Translate needs internet. LLM needs a configured service/model.\n"
            "A DirectX 12-compatible GPU and its driver are required. No CPU inference fallback.\n"
            "Settings/logs are stored in %LOCALAPPDATA%\\MangaTranslatorServer.\n"
            "Network access is off by default. Enable it only on a trusted network.\n",
            encoding="utf-8")
        total = sum(manifest.values()) / 1024**2
        print(f"Built: {OUTPUT}\nBundled runtime: {total:.1f} MiB. Models and local settings excluded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        raise SystemExit(1)

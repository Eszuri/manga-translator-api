#!/usr/bin/env python3
import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path


def find_browser():
    custom = os.environ.get("CHROME_PATH")
    if custom and os.path.isfile(custom):
        return custom

    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    for name in ["chrome", "google-chrome", "chromium", "msedge", "brave"]:
        found = shutil.which(name)
        if found:
            return found
    return None


def build():
    root_dir = Path(__file__).resolve().parents[2]
    ext_dir = root_dir / "extension"
    dist_dir = root_dir / "dist"

    if not (ext_dir / "manifest.json").is_file():
        print(f"Error: manifest.json tidak ditemukan di {ext_dir}")
        sys.exit(1)

    print("Memulai build extension...")
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build .zip
    zip_path = dist_dir / "manga-translator.zip"
    ignore = {".git", ".DS_Store", "Thumbs.db", "*.pyc", "__pycache__"}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in ext_dir.rglob("*"):
            if file.is_file() and not any(file.match(p) for p in ignore):
                zf.write(file, file.relative_to(ext_dir))

    # 2. Build .crx
    browser = find_browser()
    pem_path = dist_dir / "manga-translator.pem"
    crx_path = dist_dir / "manga-translator.crx"

    if browser:
        cmd = [browser, f"--pack-extension={ext_dir}", "--no-message-box"]
        if pem_path.is_file():
            cmd.append(f"--pack-extension-key={pem_path}")

        subprocess.run(cmd, capture_output=True, text=True)

        generated_crx = root_dir / "extension.crx"
        generated_pem = root_dir / "extension.pem"

        if generated_pem.is_file() and not pem_path.is_file():
            shutil.move(str(generated_pem), str(pem_path))
        elif generated_pem.is_file():
            generated_pem.unlink(missing_ok=True)

        if generated_crx.is_file():
            shutil.move(str(generated_crx), str(crx_path))

    print("\nBuild selesai:")
    if crx_path.is_file():
        print(f"- {crx_path.relative_to(root_dir)}")
    print(f"- {zip_path.relative_to(root_dir)}")


if __name__ == "__main__":
    build()

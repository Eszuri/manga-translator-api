@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
pushd "%~dp0\.." || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Lingkungan .venv tidak ditemukan.
    pause
    exit /b 1
)

.\.venv\Scripts\python.exe scripts\manga_tui.py %*
popd

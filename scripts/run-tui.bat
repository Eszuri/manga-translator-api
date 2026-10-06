@echo off
setlocal
chcp 65001 >nul
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
pushd "%~dp0\.." || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Lingkungan .venv tidak ditemukan.
    pause
    exit /b 1
)

.\.venv\Scripts\python.exe scripts\manga_tui.py %*
popd

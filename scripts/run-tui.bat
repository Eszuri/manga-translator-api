@echo off
setlocal
chcp 65001 >nul
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
pushd "%~dp0\.."
if errorlevel 1 (
    echo [ERROR] Project directory could not be opened.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv environment not found.
    popd
    pause
    exit /b 1
)

.\.venv\Scripts\python.exe scripts\manga_tui.py %*
set "RUN_EXIT_CODE=%ERRORLEVEL%"
popd
pause
exit /b %RUN_EXIT_CODE%

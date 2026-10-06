@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
pushd "%~dp0\.." || (
    echo [FAILED] Project root could not be opened.
    set "SERVER_EXIT_CODE=1"
    goto :finish
)

if not exist ".venv\Scripts\python.exe" (
    echo [FAILED] Local Python environment is missing: .venv
    popd
    set "SERVER_EXIT_CODE=1"
    goto :finish
)

echo Starting local API. Press Ctrl+C to stop.
pushd api || (
    set "SERVER_EXIT_CODE=1"
    popd
    goto :finish
)
..\.venv\Scripts\python.exe -m tools.run_local
set "SERVER_EXIT_CODE=%ERRORLEVEL%"
popd
popd
:finish
pause
exit /b %SERVER_EXIT_CODE%

@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
pushd "%~dp0\..\..\.." || exit /b 1

if not exist ".venv-gpu\Scripts\python.exe" (
    echo [FAILED] Local GPU environment is missing: .venv-gpu
    popd
    exit /b 1
)

echo Starting local API. Press Ctrl+C to stop.
pushd api || exit /b 1
..\.venv-gpu\Scripts\python.exe -m tools.run_local
set "SERVER_EXIT_CODE=%ERRORLEVEL%"
popd
popd
exit /b %SERVER_EXIT_CODE%

@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
set "PROCESS_EXIT_CODE=1"

pushd "%~dp0.." || (
    echo [FAILED] Project root could not be opened.
    goto :finish
)

set "SOURCE_DIR=api\Images\original Images"
if not exist "%SOURCE_DIR%\." (
    echo [FAILED] Source image folder does not exist: %SOURCE_DIR%
    goto :leave_root
)

set "HAS_IMAGE="
for %%E in (jpg jpeg png webp) do if exist "%SOURCE_DIR%\*.%%E" set "HAS_IMAGE=1"
if not defined HAS_IMAGE (
    echo [FAILED] No JPEG, PNG, or WebP images found in: %SOURCE_DIR%
    goto :leave_root
)

if not exist ".venv-gpu\Scripts\python.exe" (
    echo [FAILED] Local GPU environment is missing: .venv-gpu
    goto :leave_root
)

pushd api || (
    echo [FAILED] API folder could not be opened.
    goto :leave_root
)

echo Processing images from Images\original Images...
..\.venv-gpu\Scripts\python.exe -m tools.build_local --device gpu %*
set "PROCESS_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%PROCESS_EXIT_CODE%"=="0" echo [FAILED] Image processing failed.
:leave_root
popd
:finish
pause
exit /b %PROCESS_EXIT_CODE%

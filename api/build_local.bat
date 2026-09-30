@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

"..\.venv-gpu\Scripts\python.exe" build_local.py --device gpu %*
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%BUILD_EXIT_CODE%"=="0" (
    echo [FAILED] Local image build failed. Check the build manifest.
    exit /b %BUILD_EXIT_CODE%
)
echo [SUCCESS] Local image build completed.

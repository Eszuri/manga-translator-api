@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

"..\.venv-gpu\Scripts\python.exe" pipeline_validation.py --device gpu %*
set "PIPELINE_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%PIPELINE_EXIT_CODE%"=="0" (
    echo [FAILED] End-to-end manga test failed. Check the run manifest.
    exit /b %PIPELINE_EXIT_CODE%
)
echo [SUCCESS] End-to-end manga test completed.

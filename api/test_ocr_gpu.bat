@echo off
REM ========================================================
REM MANGA OCR TEST - GPU (DirectML)
REM ========================================================
echo ========================================================
echo  RUNNING DETECTION (HYBRID) + MANGA OCR WITH GPU
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_ocr.py --limit 80 --detector hybrid --device gpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] OCR test on GPU encountered an error.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] OCR test on GPU completed successfully!
if "%~1"=="" pause

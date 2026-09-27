@echo off
REM ========================================================
REM MANGA OCR TEST - CPU
REM ========================================================
echo ========================================================
echo  RUNNING DETECTION (HYBRID) + MANGA OCR WITH CPU
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_ocr.py --device cpu --detector hybrid %*
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] OCR test on CPU encountered an error.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] OCR test on CPU completed successfully!
if "%~1"=="" pause

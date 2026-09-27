@echo off
REM ========================================================
REM MANGA INPAINTING & TYPESETTING TEST - CPU
REM ========================================================
echo ========================================================
echo  RUNNING INPAINT & TYPESETTING PIPELINE WITH CPU (HYBRID)
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_inpaint.py --limit 5 --detector hybrid --device cpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Inpainting and typesetting pipeline test on CPU encountered an error.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Inpainting and typesetting pipeline test on CPU completed successfully!
if "%~1"=="" pause

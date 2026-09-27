@echo off
REM ========================================================
REM MANGA INPAINTING & TYPESETTING TEST - GPU (DirectML)
REM ========================================================
echo ========================================================
echo  RUNNING INPAINT & TYPESETTING PIPELINE WITH GPU (HYBRID)
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_inpaint.py --limit 80 --detector hybrid --device gpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Inpainting and typesetting pipeline test on GPU encountered an error.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Inpainting and typesetting pipeline test on GPU completed successfully!
if "%~1"=="" pause

@echo off
REM ========================================================
REM MANGA TRANSLATION TEST - GPU (DirectML)
REM ========================================================
echo ========================================================
echo  RUNNING FULL TRANSLATION PIPELINE WITH GPU (HYBRID)
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_translate.py --limit 80 --detector hybrid --device gpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Translation pipeline test on GPU encountered an error.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Translation pipeline test on GPU completed successfully!
if "%~1"=="" pause

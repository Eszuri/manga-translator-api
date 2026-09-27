@echo off
REM ========================================================
REM MANGA TRANSLATION TEST - CPU
REM ========================================================
echo ========================================================
echo  RUNNING FULL TRANSLATION PIPELINE WITH CPU (HYBRID)
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_translate.py --detector hybrid --device cpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Translation pipeline test on CPU encountered an error.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Translation pipeline test on CPU completed successfully!
if "%~1"=="" pause

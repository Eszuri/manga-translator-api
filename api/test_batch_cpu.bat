@echo off
REM ========================================================
REM BATCH FOLDER TEST - CPU (Hybrid AI + Balloon)
REM ========================================================
echo ========================================================
echo  RUNNING BATCH FOLDER TEST WITH CPU (HYBRID)
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_folder.py --detector hybrid --device cpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Batch folder test on CPU encountered an error.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Batch folder test on CPU completed successfully!
if "%~1"=="" pause

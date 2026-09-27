@echo off
REM ========================================================
REM BATCH FOLDER TEST - GPU (DirectML)
REM ========================================================
echo ========================================================
echo  RUNNING BATCH FOLDER TEST WITH GPU (HYBRID)
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_folder.py --detector hybrid --device gpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Batch folder test on GPU encountered an error.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Batch folder test on GPU completed successfully!
if "%~1"=="" pause
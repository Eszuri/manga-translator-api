@echo off
REM ========================================================
REM BATCH FOLDER TEST - KHUSUS GPU (DirectML)
REM ========================================================
echo ========================================================
echo  MENJALANKAN BATCH FOLDER TEST DENGAN GPU (DirectML)
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_folder.py --detector hybrid --device gpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [GAGAL] Batch test folder GPU mengalami kendala.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUKSES] Batch folder test GPU selesai dan berhasil!
pause
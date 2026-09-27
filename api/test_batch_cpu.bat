@echo off
REM ========================================================
REM BATCH FOLDER TEST - KHUSUS CPU (OpenCV Heuristic)
REM ========================================================
echo ========================================================
echo  MENJALANKAN BATCH FOLDER TEST DENGAN CPU
echo ========================================================

python batch_test_folder.py --detector hybrid --device cpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [GAGAL] Batch test folder CPU mengalami kendala.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUKSES] Batch folder test CPU selesai dan berhasil!
pause

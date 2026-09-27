@echo off
REM ========================================================
REM PENGUJIAN MANGA OCR KHUSUS CPU
REM ========================================================
echo ========================================================
echo  MENJALANKAN DETEKSI + MANGA OCR DENGAN CPU
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_ocr.py --device cpu --detector hybrid %*
if %ERRORLEVEL% NEQ 0 (
    echo [GAGAL] Pengujian OCR CPU mengalami kendala.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUKSES] Pengujian OCR CPU selesai dan berhasil!
if "%~1"=="" pause

@echo off
REM ========================================================
REM PENGUJIAN MANGA OCR KHUSUS GPU (DirectML)
REM ========================================================
echo ========================================================
echo  MENJALANKAN DETEKSI + MANGA OCR DENGAN GPU (DirectML)
echo ========================================================

"..\.venv-gpu\Scripts\python.exe" batch_test_ocr.py --limit 80 --detector hybrid --device gpu %*
if %ERRORLEVEL% NEQ 0 (
    echo [GAGAL] Pengujian OCR GPU mengalami kendala.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUKSES] Pengujian OCR GPU selesai dan berhasil!
if "%~1"=="" pause

@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
set "BUILD_EXIT_CODE=1"

pushd "%~dp0.." || (
    echo [FAILED] Project root could not be opened.
    goto :finish
)
if not exist ".venv-gpu\Scripts\python.exe" (
    echo [FAILED] Local GPU environment is missing: .venv-gpu
    goto :leave_root
)

.venv-gpu\Scripts\python.exe -c "from importlib.metadata import version; assert version('Nuitka') == '4.2.2'" >nul 2>&1
if errorlevel 1 (
    .venv-gpu\Scripts\python.exe -m pip install -r api\requirements\build-cli.txt
    if errorlevel 1 goto :leave_root
)

set "PYTHONPATH=%CD%\api"
.venv-gpu\Scripts\python.exe -m nuitka ^
    --mode=standalone ^
    --mingw64 ^
    --assume-yes-for-downloads ^
    --windows-console-mode=force ^
    --output-dir=dist ^
    --output-filename=MangaTranslatorAPI.exe ^
    --remove-output ^
    --jobs=4 ^
    --disable-plugins=transformers ^
    --user-plugin=api\tools\nuitka_ocr_plugin.py ^
    --include-package=app ^
    --include-package=uvicorn ^
    --include-package=huggingface_hub.utils ^
    --include-module=transformers.models.vit.image_processing_pil_vit ^
    --include-module=transformers.models.bert_japanese.tokenization_bert_japanese ^
    --include-package-data=app ^
    --include-package-data=transformers ^
    --include-package-data=unidic_lite ^
    --include-package-data=certifi ^
    api\tools\cli_api.py
if errorlevel 1 goto :leave_root

if not exist "dist\cli_api.dist\MangaTranslatorAPI.exe" (
    echo [FAILED] Standalone executable was not created.
    goto :leave_root
)

rem Nuitka package-data discovery excludes these required MeCab binary tables.
for %%F in (char.bin matrix.bin) do (
    copy /y ".venv-gpu\Lib\site-packages\unidic_lite\dicdir\%%F" "dist\cli_api.dist\unidic_lite\dicdir\%%F" >nul
    if errorlevel 1 (
        echo [FAILED] Could not package MeCab dictionary table: %%F
        goto :leave_root
    )
)

powershell.exe -NoProfile -Command "$ErrorActionPreference = 'Stop'; $files = Get-ChildItem -LiteralPath 'dist\cli_api.dist' -Recurse -File | Where-Object { $_.Extension -in @('.py', '.pyc', '.onnx') -or $_.Name -eq '.env' }; if ($files) { $files.FullName; exit 1 }"
if errorlevel 1 (
    echo [FAILED] Package contains source, bytecode, model, or .env files.
    goto :leave_root
)

"dist\cli_api.dist\MangaTranslatorAPI.exe" --check
if errorlevel 1 goto :leave_root

echo [SUCCESS] CLI API package: dist\cli_api.dist\MangaTranslatorAPI.exe
set "BUILD_EXIT_CODE=0"
:leave_root
popd
:finish
pause
exit /b %BUILD_EXIT_CODE%

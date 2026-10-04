@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
set "GUI_EXIT_CODE=1"
pushd "%~dp0\..\..\.." || goto :finish
if not exist ".venv-gpu\Scripts\python.exe" (
    echo [FAILED] Prepare .venv-gpu with api\requirements\local.txt first.
    goto :return
)
.venv-gpu\Scripts\python.exe -m pip install --timeout 180 --retries 5 -r api\requirements\build-gui.txt
if errorlevel 1 goto :return
.venv-gpu\Scripts\python.exe scripts\packaging\package_server.py
set "GUI_EXIT_CODE=%ERRORLEVEL%"
:return
popd
:finish
pause
exit /b %GUI_EXIT_CODE%

@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
set "GUI_EXIT_CODE=1"
pushd "%~dp0\..\..\.." || goto :finish
if not exist ".venv-gpu\Scripts\pythonw.exe" (
    echo [FAILED] Prepare .venv-gpu with api\requirements\local.txt first.
    goto :return
)
.venv-gpu\Scripts\python.exe -m pip install --timeout 180 --retries 5 -r api\requirements\gui.txt
if errorlevel 1 goto :return
pushd api || goto :return
..\.venv-gpu\Scripts\pythonw.exe -m tools.run_gui
set "GUI_EXIT_CODE=%ERRORLEVEL%"
popd
:return
popd
:finish
pause
exit /b %GUI_EXIT_CODE%

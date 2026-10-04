@echo off
setlocal
set "GUI_EXIT_CODE=1"
pushd "%~dp0\..\..\.." || goto :finish
if not exist "dist\MangaTranslatorServer\MangaTranslatorServer.exe" (
    echo [FAILED] Run scripts\windows\build\package-server-gui.bat first.
    goto :return
)
"dist\MangaTranslatorServer\MangaTranslatorServer.exe"
set "GUI_EXIT_CODE=%ERRORLEVEL%"
:return
popd
:finish
pause
exit /b %GUI_EXIT_CODE%

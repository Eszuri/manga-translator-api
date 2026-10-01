@echo off
setlocal
pushd "%~dp0\..\..\.." || (
    echo [FAILED] Project root could not be opened.
    set "DOCKER_EXIT_CODE=1"
    goto :finish
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\common\check-docker.ps1"
if errorlevel 1 (
    set "DOCKER_EXIT_CODE=1"
    popd
    goto :finish
)

docker compose -f compose.yaml -f compose.dev.yaml down
set "DOCKER_EXIT_CODE=%ERRORLEVEL%"
popd
:finish
pause
exit /b %DOCKER_EXIT_CODE%

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

docker compose -f compose.runtime.yaml build runtime
set "DOCKER_EXIT_CODE=%ERRORLEVEL%"
if "%DOCKER_EXIT_CODE%"=="0" echo [SUCCESS] GPU runtime image is ready.
popd
:finish
pause
exit /b %DOCKER_EXIT_CODE%

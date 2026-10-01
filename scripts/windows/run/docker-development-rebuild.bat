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

docker image inspect manga-translator-runtime:gpu >nul 2>&1 || docker compose -f compose.runtime.yaml build runtime
if errorlevel 1 (
    set "DOCKER_EXIT_CODE=1"
    popd
    goto :finish
)

docker compose -f compose.yaml -f compose.dev.yaml up -d --build --wait --wait-timeout 180
set "DOCKER_EXIT_CODE=%ERRORLEVEL%"
if "%DOCKER_EXIT_CODE%"=="0" echo [SUCCESS] Docker development API was rebuilt and started.
popd
:finish
pause
exit /b %DOCKER_EXIT_CODE%

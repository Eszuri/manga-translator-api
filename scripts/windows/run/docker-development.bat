@echo off
setlocal
pushd "%~dp0\..\..\.." || exit /b 1

docker compose -f compose.yaml -f compose.dev.yaml up -d --no-build
set "DOCKER_EXIT_CODE=%ERRORLEVEL%"
if "%DOCKER_EXIT_CODE%"=="0" echo [SUCCESS] Docker development API is running with auto-reload.
popd
exit /b %DOCKER_EXIT_CODE%

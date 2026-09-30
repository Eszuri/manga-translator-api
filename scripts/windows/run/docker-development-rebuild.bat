@echo off
setlocal
pushd "%~dp0\..\..\.." || exit /b 1

docker compose -f compose.yaml -f compose.dev.yaml up -d --build
set "DOCKER_EXIT_CODE=%ERRORLEVEL%"
if "%DOCKER_EXIT_CODE%"=="0" echo [SUCCESS] Docker development API was rebuilt and started.
popd
exit /b %DOCKER_EXIT_CODE%

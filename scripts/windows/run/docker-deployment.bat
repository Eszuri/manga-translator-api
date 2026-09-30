@echo off
setlocal
pushd "%~dp0\..\..\.." || exit /b 1

docker compose -f compose.yaml up -d --build
set "DOCKER_EXIT_CODE=%ERRORLEVEL%"
if "%DOCKER_EXIT_CODE%"=="0" echo [SUCCESS] Docker deployment API is running.
popd
exit /b %DOCKER_EXIT_CODE%

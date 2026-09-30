@echo off
setlocal
pushd "%~dp0\..\..\.." || exit /b 1

docker compose -f compose.yaml -f compose.dev.yaml down
set "DOCKER_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %DOCKER_EXIT_CODE%

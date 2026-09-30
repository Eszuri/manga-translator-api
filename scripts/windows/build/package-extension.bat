@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
pushd "%~dp0\..\..\.." || exit /b 1

python scripts\tasks\package_extension.py
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%BUILD_EXIT_CODE%"=="0" exit /b %BUILD_EXIT_CODE%
echo [SUCCESS] Extension package is available in dist.

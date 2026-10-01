@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
pushd "%~dp0\..\..\.." || (
    echo [FAILED] Project root could not be opened.
    set "BUILD_EXIT_CODE=1"
    goto :finish
)

python scripts\tasks\package_extension.py
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%BUILD_EXIT_CODE%"=="0" (
    echo [FAILED] Extension package build failed.
    goto :finish
)

echo [SUCCESS] Extension package is available in dist.
:finish
pause
exit /b %BUILD_EXIT_CODE%

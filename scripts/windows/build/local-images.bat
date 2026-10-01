@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
pushd "%~dp0\..\..\.." || (
    echo [FAILED] Project root could not be opened.
    set "BUILD_EXIT_CODE=1"
    goto :finish
)

if not exist ".venv-gpu\Scripts\python.exe" (
    echo [FAILED] Local GPU environment is missing: .venv-gpu
    popd
    set "BUILD_EXIT_CODE=1"
    goto :finish
)

pushd api || (
    set "BUILD_EXIT_CODE=1"
    popd
    goto :finish
)
..\.venv-gpu\Scripts\python.exe -m tools.build_local --device gpu %*
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
popd
popd

if not "%BUILD_EXIT_CODE%"=="0" (
    echo [FAILED] Local image build failed. Check api\Images\build Images.
    goto :finish
)

echo [SUCCESS] Local image build completed.
:finish
pause
exit /b %BUILD_EXIT_CODE%

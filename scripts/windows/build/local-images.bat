@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
pushd "%~dp0\..\..\.." || exit /b 1

if not exist ".venv-gpu\Scripts\python.exe" (
    echo [FAILED] Local GPU environment is missing: .venv-gpu
    popd
    exit /b 1
)

pushd api || exit /b 1
..\.venv-gpu\Scripts\python.exe -m tools.build_local --device gpu %*
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
popd
popd

if not "%BUILD_EXIT_CODE%"=="0" (
    echo [FAILED] Local image build failed. Check api\test-data\builds.
    exit /b %BUILD_EXIT_CODE%
)

echo [SUCCESS] Local image build completed.

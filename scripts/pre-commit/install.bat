@echo off
setlocal

set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..\..") do set REPO_ROOT=%%~fI
set CONFIG_PATH=%REPO_ROOT%\.pre-commit-config.yaml

cd /d "%REPO_ROOT%"

where python >nul 2>nul
if errorlevel 1 (
    echo python is required but was not found.
    exit /b 1
)

if not exist "%CONFIG_PATH%" (
    echo .pre-commit-config.yaml was not found at %CONFIG_PATH%.
    exit /b 1
)

python -m pre_commit --version >nul 2>nul
if errorlevel 1 (
    echo Installing pre-commit...
    python -m pip install --user pre-commit
    if errorlevel 1 exit /b 1
)

echo Installing git pre-commit hook...
python -m pre_commit install --hook-type pre-commit --config "%CONFIG_PATH%"
if errorlevel 1 exit /b 1

echo Installing hook environments and dependencies from .pre-commit-config.yaml...
python -m pre_commit install-hooks --config "%CONFIG_PATH%"
if errorlevel 1 exit /b 1

echo pre-commit is ready.
echo You can verify it with: python -m pre_commit run --all-files
exit /b 0

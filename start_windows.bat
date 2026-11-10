@echo off
REM HilbertFuse one-click start for Windows.
REM   Double-click this file  -> first run installs everything (5-10 min, internet needed), then shows a menu.
REM   Or from a terminal:  start_windows.bat predict --model models\golub --input my.csv --out out.csv
setlocal
cd /d "%~dp0"
title HilbertFuse
if not exist ".venv\.hilbertfuse_ready" (
  echo First run: setting up. This takes 5-10 minutes and needs internet...
  where python >nul 2>nul
  if errorlevel 1 (
    echo.
    echo ERROR: Python was not found. Install Python 3.10, 3.11 or 3.12 from https://www.python.org/downloads/
    echo        and tick "Add python.exe to PATH", then double-click this file again.
    pause
    exit /b 1
  )
  python -m venv .venv || goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
  ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :fail
  ".venv\Scripts\python.exe" -m pip install -e ".[datasets]" || goto :fail
  echo ok> ".venv\.hilbertfuse_ready"
  echo Setup finished.
)
if "%~1"=="" (
  ".venv\Scripts\python.exe" -m hilbertfuse.cli menu
) else (
  ".venv\Scripts\python.exe" -m hilbertfuse.cli %*
)
echo.
pause
exit /b 0

:fail
echo.
echo Setup failed. Scroll up to read the error. Usual fixes: use Python 3.10-3.12 (not 3.13), check your internet connection, run again.
pause
exit /b 1

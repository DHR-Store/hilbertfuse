@echo off
REM Data preparation launcher for Windows.  Examples (from the project folder):
REM   dataprep\prepare_windows.bat check --input my_raw.csv --label-col label
REM   dataprep\prepare_windows.bat fit   --input my_raw.csv --label-col label --out clean_data
REM   dataprep\prepare_windows.bat apply --prep clean_data\prep_model.joblib --input new_raw.csv --out new_clean.csv
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (set PY=".venv\Scripts\python.exe") else (set PY=python)
%PY% dataprep\prepare_data.py %*
echo.
pause

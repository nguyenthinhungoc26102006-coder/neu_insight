@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo =================================================
echo  NEU GRADE INTELLIGENCE - FIT INNOVATION 2026
echo =================================================
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating Python virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create environment. Install Python 3.11+ first.
    pause
    exit /b 1
  )
)
echo [2/3] Installing / verifying dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)
echo [3/3] Starting localhost:8000 (private, only this computer)...
start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause

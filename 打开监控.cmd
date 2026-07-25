@echo off
setlocal
cd /d "%~dp0"
if not defined KARDS_RUNS_DIR set "KARDS_RUNS_DIR=D:\KardsAI\runs"
powershell.exe -NoProfile -WindowStyle Hidden -Command "$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue; if (-not $listener) { Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'monitor.py' -WorkingDirectory (Get-Location) -WindowStyle Hidden }"
timeout /t 1 /nobreak >nul
start "KARDS AI Monitor" "http://127.0.0.1:8765"

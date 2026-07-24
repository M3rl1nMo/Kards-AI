@echo off
setlocal
cd /d C:\Users\ASUS\OneDrive\MicrosoftDocuments\Kards-AI
powershell.exe -NoProfile -WindowStyle Hidden -Command "$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue; if (-not $listener) { Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'monitor.py' -WorkingDirectory (Get-Location) -WindowStyle Hidden }"
timeout /t 1 /nobreak >nul
start "KARDS AI Monitor" "http://127.0.0.1:8765"

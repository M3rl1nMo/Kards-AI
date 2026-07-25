@echo off
setlocal
call "%~dp0training\d_drive_env.cmd"
powershell.exe -NoProfile -WindowStyle Hidden -Command "$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue; if (-not $listener) { Start-Process -FilePath '%KARDS_VENV%\Scripts\python.exe' -ArgumentList 'monitor.py' -WorkingDirectory '%KARDS_WORKSPACE%' -WindowStyle Hidden }"
timeout /t 1 /nobreak >nul
start "KARDS AI Monitor" "http://127.0.0.1:8765"

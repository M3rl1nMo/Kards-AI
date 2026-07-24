@echo off
setlocal
schtasks.exe /Query /TN "KardsAI-Monitor" >nul 2>&1
if errorlevel 1 goto :open
schtasks.exe /Run /TN "KardsAI-Monitor" >nul 2>&1
:open
start "KARDS AI Monitor" "http://127.0.0.1:8765"

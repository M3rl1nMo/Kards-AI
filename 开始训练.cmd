@echo off
setlocal
schtasks.exe /Query /TN "KardsAI-Parallel-Selfplay" >nul 2>&1
if errorlevel 1 goto :missing
schtasks.exe /Run /TN "KardsAI-Parallel-Selfplay"
if errorlevel 1 goto :not_started
echo CUDA self-play training started.
echo Monitor: http://127.0.0.1:8765
pause
exit /b 0
:missing
echo Training task is missing. Run the Codex setup first.
pause
exit /b 1
:not_started
echo Training may already be running, or task start failed.
pause
exit /b 1

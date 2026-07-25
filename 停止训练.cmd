@echo off
setlocal
if not defined KARDS_RUNS_DIR set "KARDS_RUNS_DIR=D:\KardsAI\runs"
if not exist "%KARDS_RUNS_DIR%" mkdir "%KARDS_RUNS_DIR%"
type nul > "%KARDS_RUNS_DIR%\STOP"
echo Graceful stop requested. Current phase will finish and save its result.
pause
exit /b 0

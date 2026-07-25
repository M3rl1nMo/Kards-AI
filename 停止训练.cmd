@echo off
setlocal
call "%~dp0training\d_drive_env.cmd"
type nul > "%KARDS_RUNS_DIR%\STOP"
echo Graceful stop requested. Current phase will finish and save its result.
pause
exit /b 0

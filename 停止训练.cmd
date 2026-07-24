@echo off
setlocal
type nul > runs\STOP
echo Graceful stop requested. Current phase will finish and save its result.
pause
exit /b 0

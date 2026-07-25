@echo off
setlocal
call "%~dp0d_drive_env.cmd"
cd /d "%KARDS_WORKSPACE%"
"%KARDS_VENV%\Scripts\python.exe" monitor.py >> "%KARDS_RUNS_DIR%\monitor.log" 2>> "%KARDS_RUNS_DIR%\monitor.err.log"

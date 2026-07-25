@echo off
setlocal
call "%~dp0training\d_drive_env.cmd"
if not exist "%KARDS_WORKSPACE%\training\run_parallel_selfplay.cmd" (
  echo D: training workspace is not prepared.
  echo Run "%~dp0training\setup_d_drive_training.cmd" once first.
  pause
  exit /b 1
)
start "KARDS AI Training" /D "%KARDS_WORKSPACE%" cmd /c "call training\run_parallel_selfplay.cmd"
echo CUDA self-play training started from D:.
echo Runtime data and caches: %KARDS_HOME%
echo Monitor: http://127.0.0.1:8765
pause
exit /b 0

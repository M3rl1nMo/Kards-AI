@echo off
setlocal
call "%~dp0d_drive_env.cmd"
if not exist "%KARDS_WORKSPACE%\auto_train.py" (
  echo D: training workspace is missing. Run training\setup_d_drive_training.cmd first.
  exit /b 1
)
if not exist "%KARDS_VENV%\Scripts\python.exe" (
  echo D: virtual environment is missing. Run training\setup_d_drive_training.cmd first.
  exit /b 1
)
cd /d "%KARDS_WORKSPACE%"
"%KARDS_VENV%\Scripts\python.exe" auto_train.py --cycles 0 --episodes 32 --mcts-simulations 64 --workers 4 --updates 100 --batch-size 256 --evaluation-games 50 >> "%KARDS_RUNS_DIR%\auto_train.log" 2>> "%KARDS_RUNS_DIR%\auto_train.err.log"

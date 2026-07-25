@echo off
setlocal
cd /d "%~dp0.."
if not defined KARDS_RUNS_DIR set "KARDS_RUNS_DIR=D:\KardsAI\runs"
if not exist "%KARDS_RUNS_DIR%" mkdir "%KARDS_RUNS_DIR%"
.venv\Scripts\python.exe auto_train.py --cycles 0 --episodes 32 --mcts-simulations 64 --workers 4 --updates 100 --batch-size 256 --evaluation-games 50 >> "%KARDS_RUNS_DIR%\auto_train.log" 2>> "%KARDS_RUNS_DIR%\auto_train.err.log"

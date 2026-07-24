@echo off
cd /d C:\Users\ASUS\OneDrive\MicrosoftDocuments\Kards-AI
.venv\Scripts\python.exe auto_train.py --cycles 0 --episodes 32 --mcts-simulations 64 --workers 1 --updates 100 --batch-size 64 --evaluation-games 50 >> runs\auto_train.log 2>> runs\auto_train.err.log

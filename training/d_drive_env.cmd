@echo off
rem Shared environment: keep every mutable training artifact on D:.
set "KARDS_HOME=D:\KardsAI"
set "KARDS_WORKSPACE=%KARDS_HOME%\workspace"
set "KARDS_PYTHON=%KARDS_HOME%\python\python.exe"
set "KARDS_VENV=%KARDS_HOME%\training-venv-py312"
set "KARDS_RUNS_DIR=%KARDS_HOME%\runs"
set "TEMP=%KARDS_HOME%\temp"
set "TMP=%KARDS_HOME%\temp"
set "PYTHONPYCACHEPREFIX=%KARDS_HOME%\pycache"
set "PIP_CACHE_DIR=%KARDS_HOME%\pip-cache"
set "TORCH_HOME=%KARDS_HOME%\torch-cache"
set "TORCHINDUCTOR_CACHE_DIR=%KARDS_HOME%\torchinductor-cache"
set "CUDA_CACHE_PATH=%KARDS_HOME%\cuda-cache"
for %%D in ("%KARDS_HOME%" "%KARDS_RUNS_DIR%" "%TEMP%" "%PYTHONPYCACHEPREFIX%" "%PIP_CACHE_DIR%" "%TORCH_HOME%" "%TORCHINDUCTOR_CACHE_DIR%" "%CUDA_CACHE_PATH%") do if not exist %%~D mkdir %%~D

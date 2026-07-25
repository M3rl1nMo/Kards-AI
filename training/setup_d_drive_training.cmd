@echo off
setlocal
call "%~dp0d_drive_env.cmd"
set "SOURCE_REPO=%~dp0.."

if not exist "%KARDS_WORKSPACE%\.git" (
  echo Creating the D: training workspace from %SOURCE_REPO% ...
  git clone --no-hardlinks "%SOURCE_REPO%" "%KARDS_WORKSPACE%"
  if errorlevel 1 exit /b 1
) else (
  echo D: training workspace already exists: %KARDS_WORKSPACE%
)

if not exist "%KARDS_PYTHON%" (
  echo Complete D: Python is missing: %KARDS_PYTHON%
  exit /b 1
)

if not exist "%KARDS_VENV%\Scripts\python.exe" (
  echo Creating a D: virtual environment ...
  "%KARDS_PYTHON%" -m venv "%KARDS_VENV%"
  if errorlevel 1 (
    echo Failed to create the D: virtual environment.
    exit /b 1
  )
)

"%KARDS_VENV%\Scripts\python.exe" -c "import numpy, yaml, torch" >nul 2>&1
if errorlevel 1 (
  echo Installing missing CUDA training dependencies on D: ...
  "%KARDS_VENV%\Scripts\python.exe" -m pip install --upgrade pip
  "%KARDS_VENV%\Scripts\python.exe" -m pip install numpy PyYAML filelock typing-extensions "setuptools<82" sympy networkx jinja2 fsspec
  rem Install the large wheel last and without resolver churn: it is cached on D:.
  "%KARDS_VENV%\Scripts\python.exe" -m pip install --no-deps --timeout 900 --retries 10 "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 exit /b 1
)

echo.
echo D: training setup is ready.
echo Workspace: %KARDS_WORKSPACE%
echo Virtual environment: %KARDS_VENV%
echo Runtime data and caches: %KARDS_HOME%

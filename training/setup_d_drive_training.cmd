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

if not exist "%KARDS_VENV%\Scripts\python.exe" (
  echo Creating a D: virtual environment ...
  py -3 -m venv "%KARDS_VENV%"
  if errorlevel 1 (
    echo Python launcher ^(py -3^) was not found. Install Python, then rerun this script.
    exit /b 1
  )
  "%KARDS_VENV%\Scripts\python.exe" -m pip install --upgrade pip
  "%KARDS_VENV%\Scripts\python.exe" -m pip install -r "%KARDS_WORKSPACE%\requirements-training.txt"
)

echo.
echo D: training setup is ready.
echo Workspace: %KARDS_WORKSPACE%
echo Virtual environment: %KARDS_VENV%
echo Runtime data and caches: %KARDS_HOME%

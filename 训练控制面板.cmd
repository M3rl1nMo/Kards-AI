@echo off
setlocal
call "%~dp0training\d_drive_env.cmd"
if exist "%KARDS_VENV%\Scripts\pythonw.exe" (
  start "KARDS AI 训练控制" /D "%KARDS_WORKSPACE%" "%KARDS_VENV%\Scripts\pythonw.exe" tools\training_control.py
) else (
  start "KARDS AI 训练控制" /D "%KARDS_WORKSPACE%" "%KARDS_VENV%\Scripts\python.exe" tools\training_control.py
)
exit /b 0

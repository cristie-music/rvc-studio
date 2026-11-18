@echo off
rem RVC Studio: starts the local web studio and opens it in the browser (http://127.0.0.1:8791/).
rem Close this window to stop the studio.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%~dp0models\hf_home"
set "TORCH_HOME=%~dp0models	orch_home"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not exist "%~dp0python\python.exe" (
  echo RVC Studio ещё не установлена - запускаю install.bat ...
  call "%~dp0install.bat" || exit /b 1
)
title RVC Studio
"%~dp0python\python.exe" -s "%~dp0app\server.py" %*
if errorlevel 1 pause

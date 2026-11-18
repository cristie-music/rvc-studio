@echo off
rem RVC Studio without the browser: drag a track onto this file -> dataset ZIP in projects\<...>\export
rem   dataset.bat "song.wav" [--variants clean,tts,aug] [--blocks 60] [--text book.txt]
setlocal
chcp 65001 >nul
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%~dp0models\hf_home"
set "TORCH_HOME=%~dp0models	orch_home"
"%~dp0python\python.exe" -s "%~dp0app\make_dataset.py" %*
pause

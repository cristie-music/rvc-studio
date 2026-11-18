@echo off
rem Shows which device RVC Studio will use (GPU or CPU) and why.
setlocal
chcp 65001 >nul
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%~dp0models\hf_home"
set "TORCH_HOME=%~dp0models	orch_home"
"%~dp0python\python.exe" -s "%~dp0app\studio\device.py"
echo.
nvidia-smi 2>nul || echo nvidia-smi не найден: драйвер NVIDIA не установлен (или видеокарта не NVIDIA) - будет CPU.
echo.
echo Нужен драйвер NVIDIA 560+ для CUDA 12.6. Без видеокарты всё работает на процессоре (медленнее).
pause

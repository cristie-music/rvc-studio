@echo off
rem ============================================================================================
rem  RVC STUDIO - installer for Windows 10/11 (x64).
rem  Everything goes INTO this folder: portable Python 3.11, PyTorch, libraries, neural networks.
rem  Nothing is installed system-wide. Needs internet once. Uses curl + tar built into Windows.
rem
rem    install.bat            auto: NVIDIA GPU -> PyTorch CUDA 12.6, Intel Arc -> PyTorch XPU, else CPU
rem    install.bat cpu        force the CPU build (smaller download, works everywhere)
rem    install.bat cuda       force the CUDA build
rem    install.bat xpu        force the Intel XPU build
rem    set RVC_LITE=1         do not pre-download the TTS model (~4.5 GB): it is fetched on first use
rem ============================================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%~dp0models\hf_home"
set "TORCH_HOME=%~dp0models	orch_home"
set "PY=%~dp0python\python.exe"
set "PY_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20261021/cpython-3.11.17%%2B20261021-x86_64-pc-windows-msvc-install_only.tar.gz"
set "TORCH_VER=2.14.1"

echo.
echo  ===============================================================
echo    RVC STUDIO - установка
echo  ===============================================================
echo.

rem ---------- 1. portable Python
if exist "%PY%" goto have_python
echo [1/6] Портативный Python 3.11 ...
curl -L --fail --retry 3 -o python.tar.gz "%PY_URL%" || goto fail
tar -xzf python.tar.gz || goto fail
del python.tar.gz
if exist "python\Lib\EXTERNALLY-MANAGED" del "python\Lib\EXTERNALLY-MANAGED"
:have_python
echo [1/6] Python: OK
"%PY%" -s -m ensurepip --upgrade >nul 2>&1
"%PY%" -s -m pip install --no-warn-script-location --upgrade pip >nul 2>&1

rem ---------- 2. which PyTorch build
set "MODE=%~1"
if "%MODE%"=="" (
  set "MODE=cpu"
  where nvidia-smi >nul 2>&1 && (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name^,driver_version --format^=csv^,noheader 2^>nul') do (
      echo        найдена видеокарта NVIDIA: %%g
      set "MODE=cuda"
    )
  )
  if "!MODE!"=="cpu" (
    for /f "tokens=*" %%g in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name" 2^>nul') do (
      echo        видеокарта: %%g
      echo %%g | findstr /i /c:"Arc" >nul && set "MODE=xpu"
    )
  )
)
echo [2/6] Сборка PyTorch: %MODE%
if /i "%MODE%"=="cuda" set "TORCH_IDX=https://download.pytorch.org/whl/cu126"
if /i "%MODE%"=="cpu"  set "TORCH_IDX=https://download.pytorch.org/whl/cpu"
if /i "%MODE%"=="xpu"  set "TORCH_IDX=https://download.pytorch.org/whl/xpu"
if not defined TORCH_IDX (echo неизвестный режим %MODE% & goto fail)

"%PY%" -s -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('%TORCH_VER%') and (('%MODE%'!='cuda') or torch.version.cuda) else 1)" >nul 2>&1 && (
  echo [3/6] PyTorch уже установлен
  goto have_torch
)
echo [3/6] PyTorch (%MODE%) - самая большая загрузка, CUDA ~2.5 GB, CPU ~200 MB ...
set "TORCH_SPEC=torch==%TORCH_VER%"
if /i "%MODE%"=="xpu" set "TORCH_SPEC=torch"
"%PY%" -s -m pip install --no-warn-script-location "%TORCH_SPEC%" --index-url %TORCH_IDX%
if errorlevel 1 (
  echo   ! эта сборка PyTorch не установилась - ставлю CPU-версию
  "%PY%" -s -m pip install --no-warn-script-location torch==%TORCH_VER% --index-url https://download.pytorch.org/whl/cpu || goto fail
)
:have_torch

echo [4/6] Библиотеки: librosa, Qwen3-TTS, RUAccent, faster-whisper, WORLD ...
"%PY%" -s -m pip install --no-warn-script-location -r requirements.txt || goto fail
"%PY%" -s -m pip install --no-warn-script-location --no-deps qwen-tts==0.1.1 || goto fail

echo [5/6] Нейросети: разделение вокала, удаление эха/ревера, ударения, распознавание речи, озвучка ...
set "MODELS=vocals dereverb_echo dereverb karaoke ruaccent asr stress_meter w2v_ru"
if not defined RVC_LITE set "MODELS=%MODELS% tts"
"%PY%" -s app\studio\models.py %MODELS% || goto fail

echo [6/6] Проверка ...
"%PY%" -s -W ignore app\selftest.py || goto fail

if not defined RVC_NO_SHORTCUT (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$s=(New-Object -ComObject WScript.Shell); foreach($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))){ $l=$s.CreateShortcut((Join-Path $d 'RVC Studio.lnk')); $l.TargetPath='%~dp0RVCStudio.bat'; $l.WorkingDirectory='%~dp0'; $l.IconLocation='%SystemRoot%\System32\SHELL32.dll,168'; $l.Description='RVC Studio - датасет для обучения RVC из трека'; $l.Save() }" >nul 2>&1
  echo   ярлыки «RVC Studio» созданы на рабочем столе и в меню Пуск
)

echo.
echo  ===============================================================
echo    ГОТОВО. Запуск: RVCStudio.bat (или ярлык на рабочем столе)
echo  ===============================================================
if not defined RVC_NOPAUSE pause
exit /b 0

:fail
echo.
echo  УСТАНОВКА НЕ УДАЛАСЬ - см. сообщение выше (интернет? место на диске?).
echo  Можно просто запустить install.bat ещё раз: уже скачанное не качается повторно.
if not defined RVC_NOPAUSE pause
exit /b 1

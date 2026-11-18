"""Generates colab/RVCStudio_Colab.ipynb (kept as code here so the notebook stays easy to edit)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MD_INTRO = """# 🎤 RVC Studio — Google Colab

**Датасет для обучения голоса RVC из песни:** вокал отделяется от музыки, с него снимаются эхо и ревер, дальше три варианта датасета:
1. **чистый голос** из трека — без пауз, шумов и вдохов, нарезка 2–10 с с оценкой каждого куска;
2. **озвучка текста голосом певца** — Qwen3-TTS (русские ударения) читает текст, например первую главу «Гарри Поттера»; ударения ставит RUAccent, каждый блок проверяется распознаванием речи и детектором ударений meter2, неудачные переозвучиваются;
3. **копии со сдвигом высоты** — тот же голос выше/ниже, тембр сохранён.

В конце — ZIP для Applio / RVC WebUI и **рекомендуемое число эпох**.

**Перед запуском:** `Среда выполнения → Сменить среду выполнения → GPU (T4 или лучше)`. Без GPU тоже работает, но озвучка будет очень медленной.

Ссылка на студию: встроенная ссылка Colab или публичная **ngrok** (впишите `NGROK_AUTHTOKEN` в шаге 2). `HF_TOKEN` ускоряет загрузку моделей.

Шаги **1 → 2 → 3 → 4** по порядку (кнопка ▶ у каждой ячейки). Шаг 5 — то же без интерфейса."""

C_GPU = """#@title 1. Проверка GPU
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True)
print("GPU:", r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else
      "НЕТ — будет CPU (медленно). Среда выполнения → Сменить среду → T4 GPU")"""

C_CODE = r"""#@title 2. Код студии и Google Drive
#@markdown **Откуда взять код**: `upload` / `drive` / `url` — архив `rvc-studio-app.zip` (собирается `python tools\make_dist.py`), `git` — ваш репозиторий:
CODE_SOURCE = "upload"  #@param ["upload", "drive", "url", "git"]
DRIVE_ZIP = "/content/drive/MyDrive/RVCStudio/rvc-studio-app.zip"  #@param {type:"string"}
CODE_URL = ""  #@param {type:"string"}
REPO_URL = "https://github.com/USER/rvc-studio"  #@param {type:"string"}
#@markdown **Хранить проекты и нейросети на Google Drive** (не пропадут после отключения Colab, ~10 ГБ моделей не качаются заново):
USE_DRIVE_STORAGE = True  #@param {type:"boolean"}
#@markdown **Токены (необязательно).** `HF_TOKEN` — быстрее и без лимитов качает модели с Hugging Face (huggingface.co/settings/tokens).
#@markdown `NGROK_AUTHTOKEN` — открыть студию по публичной ссылке ngrok (dashboard.ngrok.com → Your Authtoken); пусто — ссылка Colab.
HF_TOKEN = "hf_YOUR_HF_TOKEN_HERE"  #@param {type:"string"}
NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"  #@param {type:"string"}
import os, shutil, zipfile, urllib.request, subprocess
RS = "/content/rvc-studio"
if USE_DRIVE_STORAGE or CODE_SOURCE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
zip_path = "/content/rvc-studio-app.zip"
if CODE_SOURCE == "git":
    shutil.rmtree(RS, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, RS], check=True)
else:
    if CODE_SOURCE == "upload":
        from google.colab import files
        print("Выберите rvc-studio-app.zip ...")
        up = files.upload()
        name = next(iter(up))
        if os.path.abspath(name) != zip_path:
            shutil.move(name, zip_path)
    elif CODE_SOURCE == "drive":
        shutil.copy(DRIVE_ZIP, zip_path)
    else:
        urllib.request.urlretrieve(CODE_URL, zip_path)
    shutil.rmtree(RS, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RS)
STORE = "/content/drive/MyDrive/RVCStudio" if USE_DRIVE_STORAGE else RS
os.makedirs(f"{STORE}/projects", exist_ok=True)
os.makedirs(f"{STORE}/models", exist_ok=True)
os.environ["RVC_PROJECTS"] = f"{STORE}/projects"
os.environ["RVC_MODELS"] = f"{STORE}/models"
def _real(tok, placeholder):
    tok = (tok or "").strip()
    return tok if tok and "YOUR_" not in tok and tok != placeholder else ""
if _real(HF_TOKEN, "hf_YOUR_HF_TOKEN_HERE"):
    os.environ["HF_TOKEN"] = _real(HF_TOKEN, "hf_YOUR_HF_TOKEN_HERE")
    print("HF_TOKEN: задан")
os.environ["RVC_NGROK_TOKEN"] = _real(NGROK_AUTHTOKEN, "YOUR_NGROK_AUTHTOKEN_HERE")
print("ngrok:", "будет использован" if os.environ["RVC_NGROK_TOKEN"] else "нет токена — ссылка Colab")
print("Код:", RS, "| проекты:", os.environ["RVC_PROJECTS"], "| нейросети:", os.environ["RVC_MODELS"])"""

C_INSTALL = """#@title 3. Установка (≈5–10 минут в первый раз, один раз за сессию)
#@markdown Отдельное окружение Python 3.11 (не трогает пакеты Colab), PyTorch под GPU Colab, нейросети (~10 ГБ; на Drive — один раз).
DOWNLOAD_TTS = True  #@param {type:"boolean"}
import os, subprocess
RS = "/content/rvc-studio"
VENV = "/content/rvc-venv"
PY = f"{VENV}/bin/python"
def sh(cmd):
    print("$", cmd[:140]); r = subprocess.run(cmd, shell=True, cwd=RS)
    if r.returncode: raise SystemExit(f"ошибка: {cmd}")
sh("pip -q install uv")
if not os.path.exists(PY):
    sh(f"uv venv -q --python 3.11 {VENV}")
gpu = subprocess.run("nvidia-smi -L", shell=True, capture_output=True).returncode == 0
idx = "https://download.pytorch.org/whl/cu126" if gpu else "https://download.pytorch.org/whl/cpu"
sh(f"uv pip install -q --python {PY} torch==2.14.1 --index-url {idx}")
sh(f"uv pip install -q --python {PY} -r requirements.txt")
sh(f"uv pip install -q --python {PY} --no-deps qwen-tts==0.1.1")
models = "vocals dereverb_echo dereverb karaoke ruaccent asr stress_meter w2v_ru" + (" tts" if DOWNLOAD_TTS else "")
sh(f"{PY} app/studio/models.py {models}")
sh(f"{PY} -W ignore app/selftest.py")
print("Установлено")"""

C_RUN = """#@title 4. Запустить студию (веб-интерфейс)
import os, subprocess, time, urllib.request
RS = "/content/rvc-studio"; PY = "/content/rvc-venv/bin/python"; PORT = 8791
if "rvc_server" in globals() and rvc_server.poll() is None:
    rvc_server.terminate(); time.sleep(1)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
log = open("/content/rvc-studio.log", "w")
rvc_server = subprocess.Popen([PY, "-u", "app/server.py", "--no-browser", "--port", str(PORT)], cwd=RS, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
for _ in range(90):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/system", timeout=2); break
    except Exception:
        time.sleep(1)
else:
    print(open("/content/rvc-studio.log").read()); raise SystemExit("сервер не запустился")
NGROK = os.environ.get("RVC_NGROK_TOKEN", "")
public = None
if NGROK:
    try:
        subprocess.run("pip -q install pyngrok", shell=True, check=True)
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = NGROK
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
        public = ngrok.connect(PORT, "http").public_url
    except Exception as e:
        print("ngrok не запустился:", e, "— использую ссылку Colab")
if public:
    from IPython.display import display, HTML
    print("Студия запущена. Публичная ссылка ngrok (работает, пока открыт этот ноутбук):")
    display(HTML(f'<a href="{public}" target="_blank" style="font-size:18px">▶ Открыть RVC Studio: {public}</a>'))
else:
    from google.colab import output
    print("Студия запущена. Откройте её по ссылке (лучше в отдельной вкладке):")
    output.serve_kernel_port_as_window(PORT, anchor_text="▶ Открыть RVC Studio")
    output.serve_kernel_port_as_iframe(PORT, height=900)"""

MD_CLI = """### 5. (необязательно) Без интерфейса
Загрузите трек — получите ZIP с датасетом и рекомендуемое число эпох."""

C_CLI = """#@title 5. Сделать датасет и скачать ZIP
VARIANTS = "clean,tts"  #@param ["clean", "clean,tts", "clean,tts,aug", "clean,aug"]
BLOCKS = 60  #@param {type:"integer"}
DEREVERB = "normal"  #@param ["off", "light", "normal", "strong"]
SAMPLE_RATE = 40000  #@param [32000, 40000, 48000]
import os, subprocess
from google.colab import files
RS = "/content/rvc-studio"; PY = "/content/rvc-venv/bin/python"
up = files.upload()
src = os.path.abspath(next(iter(up)))
cmd = [PY, "-u", "app/make_dataset.py", src, "--variants", VARIANTS, "--blocks", str(BLOCKS),
       "--dereverb", DEREVERB, "--sr", str(SAMPLE_RATE)]
env = dict(os.environ, PYTHONIOENCODING="utf-8")
p = subprocess.Popen(cmd, cwd=RS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
last = ""
for line in p.stdout:
    print(line, end=""); last = line.strip() or last
p.wait()
if p.returncode == 0 and last.endswith(".zip"):
    files.download(last)"""


def cell_md(src):
    return dict(cell_type="markdown", metadata={}, source=src.splitlines(True))


def cell_code(src):
    return dict(cell_type="code", metadata={"cellView": "form"}, execution_count=None, outputs=[],
                source=src.splitlines(True))


nb = dict(nbformat=4, nbformat_minor=0,
          metadata=dict(colab=dict(provenance=[], gpuType="T4", name="RVCStudio_Colab.ipynb"),
                        kernelspec=dict(name="python3", display_name="Python 3"), accelerator="GPU",
                        language_info=dict(name="python")),
          cells=[cell_md(MD_INTRO), cell_code(C_GPU), cell_code(C_CODE), cell_code(C_INSTALL), cell_code(C_RUN),
                 cell_md(MD_CLI), cell_code(C_CLI)])
os.makedirs(os.path.join(ROOT, "colab"), exist_ok=True)
dst = os.path.join(ROOT, "colab", "RVCStudio_Colab.ipynb")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("  ", dst)

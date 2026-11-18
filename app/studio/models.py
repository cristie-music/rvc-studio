"""Model registry + downloader. All weights live in <studio>/models (override: env RVC_MODELS).

Direct files (separation networks) are fetched with urllib from a list of mirrors.
Hugging Face repositories (TTS, ASR, stress model) are fetched with huggingface_hub (honours HF_ENDPOINT).
"""
import os
import shutil
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.environ.get("RVC_MODELS") or os.path.join(ROOT, "models")
# portable: every cache (Hugging Face, PyTorch hub) stays inside the studio folder, nothing goes to the user profile
os.environ.setdefault("HF_HOME", os.path.join(MODELS, "hf_home"))
os.environ.setdefault("TORCH_HOME", os.path.join(MODELS, "torch_home"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HF = "https://huggingface.co"

# ---------------------------------------------------------------- separation (Mel-Band RoFormer)
SEPARATION = {
    "vocals": dict(
        title="Вокал из трека · Mel-Band RoFormer (becruily)",
        about="Отделяет весь вокал от музыки. Одна из лучших открытых моделей для вокала.",
        files={"mel_band_roformer_vocals_becruily.ckpt":
               [f"{HF}/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt"],
               "config_vocals_becruily.yaml":
               [f"{HF}/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml"]},
        ckpt="mel_band_roformer_vocals_becruily.ckpt", config="config_vocals_becruily.yaml", size_mb=870),
    "karaoke": dict(
        title="Основной голос / бэк-вокал · Mel-Band RoFormer Karaoke (becruily)",
        about="Оставляет ведущий голос, убирает бэки, хоры, подпевки и даблы.",
        files={"mel_band_roformer_karaoke_becruily.ckpt":
               [f"{HF}/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt"],
               "config_karaoke_becruily.yaml":
               [f"{HF}/becruily/mel-band-roformer-karaoke/resolve/main/config_karaoke_becruily.yaml"]},
        ckpt="mel_band_roformer_karaoke_becruily.ckpt", config="config_karaoke_becruily.yaml", size_mb=870),
    "dereverb_echo": dict(
        title="Эхо / дилей + ревер · Mel-Band RoFormer v2 (Sucial)",
        about="Убирает дилей (эхо) и реверберацию с вокала. Лицензия CC BY-NC-SA 4.0 (некоммерческое использование).",
        files={"dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt":
               [f"{HF}/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt"],
               "config_dereverb_echo_mbr_v2.yaml":
               [f"{HF}/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml"]},
        ckpt="dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt", config="config_dereverb_echo_mbr_v2.yaml", size_mb=460),
    "dereverb": dict(
        title="Реверберация · Mel-Band RoFormer De-Reverb (anvuew)",
        about="Добирает остатки ревера (комнаты, плейты, хвосты). Лицензия GPL-3.0.",
        files={"dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt":
               [f"{HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"],
               "dereverb_mel_band_roformer_anvuew.yaml":
               [f"{HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml"]},
        ckpt="dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", config="dereverb_mel_band_roformer_anvuew.yaml",
        size_mb=870),
}

# ---------------------------------------------------------------- Hugging Face repositories
REPOS = {
    "tts": dict(
        title="Озвучка · Qwen3-TTS 1.7B Base + русские ударения (siriusfreak ru-stress-cf)",
        about="Клонирует голос по образцу и читает текст. Следует знакам ударения (96–100%). Apache-2.0.",
        repo="siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf", dir="qwen3-tts-ru-stress",
        allow=["*.json", "*.txt", "*.safetensors", "speech_tokenizer/*"], ignore=["lora/*", "eval/*", "meter/*", "scripts/*"],
        check="model.safetensors", size_mb=4600, marks=True),
    "tts_fast": dict(
        title="Озвучка (лёгкая) · Qwen3-TTS 0.6B Base",
        about="Меньше (для CPU и слабых видеокарт). Знаки ударения не понимает: текст идёт без них, ударения "
              "только проверяются, неверные - переозвучиваются.",
        repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base", dir="qwen3-tts-0.6b-base",
        allow=["*.json", "*.txt", "*.safetensors", "speech_tokenizer/*"], ignore=[],
        check="model.safetensors", size_mb=2500, marks=False),
    "stress_meter": dict(
        title="Проверка ударений по звуку · meter2 (siriusfreak)",
        about="Определяет, какой слог реально ударный в озвучке (98% на живой речи). Исследовательский инструмент: "
              "голова обучена в том числе на озвучке Silero (некоммерческая лицензия).",
        repo="siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf", dir="meter2",
        allow=["meter/head.pt", "meter/README.md"], ignore=[], check="meter/head.pt", size_mb=5),
    "w2v_ru": dict(
        title="Выравнивание текста по звуку · wav2vec2-large-xlsr-53-russian",
        about="Находит каждую букву текста в озвучке (для meter2). Apache-2.0.",
        repo="jonatasgrosman/wav2vec2-large-xlsr-53-russian", dir="wav2vec2-ru",
        allow=["config.json", "preprocessor_config.json", "vocab.json", "special_tokens_map.json", "pytorch_model.bin"],
        ignore=[], check="pytorch_model.bin", size_mb=1260),
    "asr": dict(
        title="Распознавание речи · Whisper large-v3-turbo (faster-whisper)",
        about="Обратная проверка озвучки: что на самом деле сказано, и где каждое слово.",
        repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo", dir="whisper-large-v3-turbo",
        allow=None, ignore=[], check="model.bin", size_mb=1620),
}

# RUAccent downloads its own files (ruaccent/accentuator) into a work dir
SPECIAL = {
    "ruaccent": dict(title="Ударения · RUAccent (turbo3.1 + словарь)",
                     about="Расставляет ударения и букву ё перед озвучкой, разбирает омографы по контексту. MIT.",
                     dir="ruaccent", size_mb=600),
}

ALL = {**SEPARATION, **REPOS, **SPECIAL}


def path(name):
    return os.path.join(MODELS, name)


def repo_dir(key):
    return path(ALL[key]["dir"])


def installed(key):
    spec = ALL[key]
    if key in SEPARATION:
        return all(os.path.exists(path(f)) for f in spec["files"])
    if key in REPOS:
        return os.path.exists(os.path.join(repo_dir(key), spec["check"]))
    if key == "ruaccent":
        d = repo_dir(key)
        return os.path.isdir(d) and any(os.path.isdir(os.path.join(d, x)) for x in ("nn", "dictionary"))
    return False


def status():
    return {k: dict(title=v["title"], about=v.get("about", ""), installed=installed(k), size_mb=v.get("size_mb"))
            for k, v in ALL.items()}


def _download_file(fn, urls, progress=None, log=print):
    dst = path(fn)
    if os.path.exists(dst):
        return
    err = None
    for url in urls:
        try:
            log(f"  загрузка {fn} <- {url.split('/')[2]}")
            headers = {"User-Agent": "rvc-studio/1.0"}
            tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if tok and url.startswith(HF):
                headers["Authorization"] = f"Bearer {tok}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, open(dst + ".part", "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    if progress:
                        progress(done, total, fn)
            shutil.move(dst + ".part", dst)
            return
        except Exception as e:  # next mirror
            err = e
            try:
                os.remove(dst + ".part")
            except OSError:
                pass
    raise RuntimeError(f"не удалось скачать {fn}: {err}")


def _dir_size(d):
    n = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            try:
                n += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return n


def download(key, progress=None, log=print):
    os.makedirs(MODELS, exist_ok=True)
    spec = ALL[key]
    if key in SEPARATION:
        for fn, urls in spec["files"].items():
            _download_file(fn, urls, progress, log)
        return
    if key in REPOS:
        from huggingface_hub import snapshot_download
        d = repo_dir(key)
        log(f"  загрузка {spec['repo']} (~{spec['size_mb']} MB) -> models/{spec['dir']}")
        import threading
        stop = threading.Event()

        def watch():   # snapshot_download has no byte callback: report the folder size instead
            total = spec["size_mb"] * 2 ** 20
            while not stop.wait(1.0):
                if progress:
                    progress(_dir_size(d), total, spec["dir"])
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            for attempt in range(3):
                try:
                    snapshot_download(spec["repo"], local_dir=d, allow_patterns=spec["allow"],
                                      ignore_patterns=spec["ignore"] or None)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"не удалось скачать {spec['repo']}: {e}")
                    log(f"  повтор загрузки ({e})")
                    time.sleep(3)
        finally:
            stop.set()
        return
    if key == "ruaccent":
        log("  загрузка RUAccent (ruaccent/accentuator) ...")
        from ruaccent import RUAccent
        acc = RUAccent()
        acc.load(omograph_model_size="turbo3.1", use_dictionary=True, workdir=repo_dir(key))
        return


def ensure(key, progress=None, log=print):
    if not installed(key):
        download(key, progress, log)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    keys = sys.argv[1:] or list(ALL)
    for k in keys:
        print(f"[{k}] {ALL[k]['title']}")

        def p(d, t, fn):
            print(f"\r    {fn}: {d / 2**20:.0f}/{t / 2**20:.0f} MB   ", end="", flush=True)
        ensure(k, p)
        print("\n    OK")

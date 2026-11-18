"""Installer check: libraries import, device detection, which models are present. Exit code 1 if something is broken."""
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ok = True
from studio import device as D  # noqa: E402

d = D.detect()
print("  устройство:", D.label(d))
if d.get("warning"):
    print("  !", d["warning"])
for mod in ("numpy", "scipy", "soundfile", "librosa", "pyworld", "yaml", "transformers", "faster_whisper", "ruaccent",
            "num2words", "huggingface_hub"):
    try:
        __import__(mod)
    except Exception as e:
        ok = False
        print(f"  ! не импортируется {mod}: {e}")
try:
    from studio import tts as TT
    TT._stub_torchaudio()
    TT._stub_sox()
    import qwen_tts  # noqa: F401
except Exception as e:
    ok = False
    print(f"  ! не импортируется qwen_tts: {e}")
try:
    import soundfile as sf
    import numpy as np
    import io
    b = io.BytesIO()
    sf.write(b, np.zeros(4410, np.float32), 44100, format="MP3")
except Exception as e:
    print(f"  ! MP3 через libsndfile недоступен ({e}): превью в браузере будет медленнее")
from studio import models as M  # noqa: E402

for k, v in M.status().items():
    print(f"  {'✓' if v['installed'] else '·'} {v['title']}")
print("  библиотеки:", "OK" if ok else "ЕСТЬ ОШИБКИ")
sys.exit(0 if ok else 1)

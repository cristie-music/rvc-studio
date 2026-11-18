"""RVC Studio without the browser: track -> dataset ZIP.

    python app/make_dataset.py "song.wav" [--variants clean,tts,aug] [--blocks 60] [--text book.txt]
                               [--dereverb normal] [--no-lead] [--device auto|cpu|gpu] [--sr 40000]
Prints the recommended RVC epochs at the end.
"""
import argparse
import os
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from studio import device as D  # noqa: E402
from studio import pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Датасет для RVC из трека")
    ap.add_argument("track")
    ap.add_argument("--variants", default="clean,tts", help="clean,tts,aug")
    ap.add_argument("--blocks", type=int, default=60, help="блоков текста для озвучки (0 = весь текст)")
    ap.add_argument("--text", help="свой текст .txt (по умолчанию harry.txt)")
    ap.add_argument("--dereverb", default="normal", choices=["off", "light", "normal", "strong"])
    ap.add_argument("--no-lead", action="store_true", help="не отделять основной голос от бэков")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sr", type=int, default=40000, choices=[32000, 40000, 48000])
    a = ap.parse_args()
    v = set(a.variants.split(","))
    pid = P.create_project(os.path.basename(a.track), lambda dst: shutil.copyfile(a.track, dst))
    if a.text:
        os.makedirs(os.path.dirname(P.text_path(pid)), exist_ok=True)
        shutil.copyfile(a.text, P.text_path(pid))
    opts = dict(device=a.device, lead_only=not a.no_lead, dereverb=a.dereverb,
                variants=dict(clean="clean" in v, tts="tts" in v, aug="aug" in v),
                clean=dict(target_sr=a.sr), tts=dict(max_blocks=a.blocks))
    job = P.Job(pid, opts)
    job.run()
    if job.status != "done":
        sys.exit(f"не получилось: {job.error}")
    import server
    man = P.load_manifest(pid)
    parts = [k for k in ("clean", "tts", "aug") if k in v]
    z = server.build_export(pid, parts)
    d = D.detect(a.device)
    t = P.recommend_training(server.summary_for(man, parts), d.get("vram_gb"), d["kind"])
    print("\n" + "=" * 64)
    print("Датасет:", z)
    if t:
        print(f"Обучение RVC: {t['epochs']} эпох (диапазон {t['epochs_low']}-{t['epochs_high']}), batch {t['batch']}, "
              f"сохранять каждые {t['save_every']}, sample rate {a.sr // 1000}k")
    print(z)


if __name__ == "__main__":
    main()

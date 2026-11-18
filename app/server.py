"""RVC Studio web server: the browser UI + JSON API around the dataset pipeline.

    python app/server.py [--host 127.0.0.1] [--port 8791] [--no-browser]

Only the standard library is used for HTTP. One heavy job runs at a time (the GPU is shared), others wait in a queue.
"""
import argparse
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

APP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)
WEB = os.path.join(APP, "web")
sys.path.insert(0, APP)

from studio import audio_io as A  # noqa: E402
from studio import device as D  # noqa: E402
from studio import models as M  # noqa: E402
from studio import pipeline as P  # noqa: E402

VERSION = "1.0.0"
os.makedirs(P.PROJECTS, exist_ok=True)
for ext, t in ((".mp3", "audio/mpeg"), (".flac", "audio/flac"), (".wav", "audio/wav"), (".js", "application/javascript"),
               (".txt", "text/plain; charset=utf-8"), (".json", "application/json")):
    mimetypes.add_type(t, ext)

JOBS = {}            # pid -> Job (latest run of that project)
QUEUE = queue.Queue()
DOWNLOADS = {}
SYSTEM = {}
ACC = {"acc": None, "lock": threading.Lock()}


def worker():
    while True:
        job = QUEUE.get()
        if job._cancel:
            job.status = "cancelled"
            continue
        try:
            job.run()
        except Exception:
            traceback.print_exc()


def system_info(refresh=False):
    if refresh or not SYSTEM:
        d = D.detect("auto")
        SYSTEM.update(d, label=D.label(d), version=VERSION, python=sys.version.split()[0],
                      colab="COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ)
    return dict(SYSTEM, models=M.status(), downloads=DOWNLOADS, queue=QUEUE.qsize(), projects_dir=P.PROJECTS,
                default_text=os.path.basename(P.DEFAULT_TEXT) if os.path.exists(P.DEFAULT_TEXT) else None,
                defaults=P.DEFAULT_OPTIONS)


def project_list():
    out = []
    for pid in sorted(os.listdir(P.PROJECTS), reverse=True):
        try:
            m = P.load_manifest(pid)
        except Exception:
            continue
        j = JOBS.get(pid)
        out.append(dict(id=pid, name=m["name"], created=m["created"], status=j.status if j else m["status"],
                        progress=j.progress if j else None, duration=m.get("duration"),
                        summary=m.get("summary") or P.dataset_summary(m)))
    return out


def safe_path(pid, rel):
    base = os.path.realpath(P.project_dir(pid))
    p = os.path.realpath(os.path.join(base, rel))
    if not (p == base or p.startswith(base + os.sep)):
        raise PermissionError(rel)
    return p


def busy(pid):
    j = JOBS.get(pid)
    return bool(j and j.status in ("queued", "running"))


def accentizer():
    with ACC["lock"]:
        if ACC["acc"] is None:
            from studio import textprep as TP
            M.ensure("ruaccent")
            ACC["acc"] = TP.load_accentizer(M.repo_dir("ruaccent"))
        return ACC["acc"]


def summary_for(man, parts=("clean", "tts", "aug")):
    s = dict(man.get("summary") or P.dataset_summary(man))
    for k in ("clean", "tts", "aug"):
        if k not in parts:
            s[f"{k}_s"] = 0
    s["total_s"] = round(s["clean_s"] + s["tts_s"] + s["aug_s"], 1)
    return s


def training_for(man, parts=("clean", "tts", "aug")):
    si = system_info()
    t = P.recommend_training(summary_for(man, parts), si.get("vram_gb"), si.get("kind"))
    if t:
        t["sample_rate"] = ((man.get("clean") or {}).get("stats") or {}).get("target_sr", 40000)
    return t


# ---------------------------------------------------------------- export
def dataset_files(pid, man, parts):
    """[(abs_path, arcname)] of every included file of the chosen variants."""
    pdir = P.project_dir(pid)
    out = []
    if "clean" in parts and man.get("clean"):
        out += [(os.path.join(pdir, "dataset", "clean", s["file"]), s["file"]) for s in man["clean"]["slices"]
                if s.get("include")]
    if "tts" in parts and (man.get("tts") or {}).get("dataset"):
        out += [(os.path.join(pdir, "dataset", "tts", s["file"]), s["file"])
                for s in man["tts"]["dataset"]["slices"] if s.get("include")]
    if "aug" in parts and man.get("aug"):
        out += [(os.path.join(pdir, "dataset", "aug", s["file"]), s["file"]) for s in man["aug"]["files"]]
    return [(a, b) for a, b in out if os.path.isfile(a)]


def readme(man, parts, n_files=None):
    s = summary_for(man, parts)
    s["files"] = n_files if n_files is not None else s["files"]
    sr = ((man.get("clean") or {}).get("stats") or {}).get("target_sr", 40000)
    L = [f"RVC STUDIO - датасет для обучения голоса «{man['name']}»", "=" * 64,
         f"Всего: {s['total_s'] / 60:.1f} мин, файлов {s['files']}, моно WAV 16 бит, {sr} Гц",
         f"  вариант 1 - чистый голос из трека (voice_*): {s['clean_s'] / 60:.1f} мин",
         f"  вариант 2 - озвучка текста голосом певца (tts_*): {s['tts_s'] / 60:.1f} мин",
         f"  вариант 3 - копии со сдвигом высоты, тембр сохранён (aug_*): {s['aug_s'] / 60:.1f} мин",
         f"В архиве: {', '.join(parts)}", "",
         "КАК ОБУЧИТЬ МОДЕЛЬ RVC",
         "  Applio: вкладка Train -> Dataset Path = распакованная папка dataset -> Sample Rate "
         f"{'40k' if sr == 40000 else '48k' if sr == 48000 else '32k' if sr == 32000 else '40k'} -> "
         "Preprocess -> Extract (f0: rmvpe) -> Train (эпохи и batch - см. ниже) -> Index.",
         "  RVC WebUI (RVC-Project): Train -> путь к папке dataset, target sample rate как выше, pitch guidance: да.",
         "  Файлы уже нарезаны (2-10 с), без пауз и шумов, громкость выровнена: дополнительная нарезка не нужна.", ""]
    t = training_for(man, parts)
    if t:
        L += ["РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ",
              f"  Эпох: {t['epochs']} (разумный диапазон {t['epochs_low']}-{t['epochs_high']}); "
              f"эффективная длина датасета {t['effective_min']} мин",
              f"  Batch size: {t['batch']} (под эту видеокарту)  ->  ~{t['steps_per_epoch']} шагов на эпоху, "
              f"~{t['total_steps']} шагов всего",
              f"  Сохранять каждые {t['save_every']} эпох, чтобы можно было взять лучшую."]
        L += [f"  * {n}" for n in t["notes"]] + [""]
    L += [
         "СОВЕТЫ",
         "  * Меньше 5 минут - модель будет нестабильной; 10-30 минут - хороший результат.",
         "  * Озвучка (tts_*) добавляет разговорную речь и редкие звуки, которых нет в песне; её полоса ограничена "
         "12 кГц (модель TTS работает на 24 кГц) - для 40k-модели это нормально.",
         "  * Копии со сдвигом высоты (aug_*) расширяют диапазон, но не заменяют настоящий голос: держите их "
         "не больше половины датасета.", ""]
    return "\n".join(L) + "\n"


def build_export(pid, parts, one_wav=False):
    man = P.load_manifest(pid)
    files = dataset_files(pid, man, parts)
    if not files:
        raise FileNotFoundError("нечего экспортировать (датасет ещё не собран?)")
    name = re.sub(r"[^\w\- ]+", "_", man["name"], flags=re.U).strip()[:60] or "voice"
    out = os.path.join(P.project_dir(pid), "export", f"RVC_dataset_{name}.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out + ".tmp", "w", allowZip64=True) as z:
        z.writestr("README_RVC_STUDIO.txt", readme(man, parts, len(files)).encode("utf-8-sig"))
        for src, arc in files:
            z.write(src, f"dataset/{arc}", compress_type=zipfile.ZIP_STORED)
        if one_wav:
            from studio import dataset as DS
            sr = ((man.get("clean") or {}).get("stats") or {}).get("target_sr", 40000)
            tmp = os.path.join(P.project_dir(pid), "export", "_all_in_one.wav")
            by_dir = {}
            for src, _ in files:
                by_dir.setdefault(os.path.dirname(src), []).append(os.path.basename(src))
            parts_wav = []
            for d, fs in by_dir.items():
                t = tmp + f".{len(parts_wav)}.wav"
                DS.concat_all(d, fs, t, sr)
                parts_wav.append(t)
            import numpy as np
            import soundfile as sf
            ys = [sf.read(t, dtype="float32")[0] for t in parts_wav]
            DS.write(tmp, np.concatenate(ys), sr)
            for t in parts_wav:
                os.remove(t)
            z.write(tmp, "all_in_one.wav", compress_type=zipfile.ZIP_STORED)
            os.remove(tmp)
    os.replace(out + ".tmp", out)
    return out


def build_folder(pid, parts):
    """export/dataset: a flat folder to point RVC / Applio at."""
    man = P.load_manifest(pid)
    files = dataset_files(pid, man, parts)
    if not files:
        raise FileNotFoundError("нечего экспортировать (датасет ещё не собран?)")
    d = os.path.join(P.project_dir(pid), "export", "dataset")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for src, arc in files:
        shutil.copyfile(src, os.path.join(d, arc))
    open(os.path.join(P.project_dir(pid), "export", "README_RVC_STUDIO.txt"), "w", encoding="utf-8-sig").write(
        readme(man, parts, len(files)))
    return d


def open_in_explorer(d):
    if os.name == "nt":
        os.startfile(d)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", d])
    else:
        subprocess.Popen(["xdg-open", d])


# ---------------------------------------------------------------- HTTP
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RVCStudio/" + VERSION

    def log_message(self, fmt, *args):
        if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
            sys.stderr.write("  http %s\n" % (fmt % args))

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def err(self, code, msg):
        self.send_json(dict(error=msg), code)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def send_file(self, path, download=False, name=None):
        if not os.path.isfile(path):
            return self.err(404, "файл не найден")
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if download:
            from urllib.parse import quote
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(name or os.path.basename(path))}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    b = f.read(min(1 << 20, left))
                    if not b:
                        break
                    self.wfile.write(b)
                    left -= len(b)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        path = unquote(u.path)
        try:
            if path in ("/", "/index.html"):
                return self.send_file(os.path.join(WEB, "index.html"))
            if path.startswith("/static/"):
                p = os.path.realpath(os.path.join(WEB, path[len("/static/"):]))
                if not p.startswith(os.path.realpath(WEB)):
                    return self.err(403, "forbidden")
                return self.send_file(p)
            if path == "/api/system":
                return self.send_json(system_info(q.get("refresh") == "1"))
            if path == "/api/projects":
                return self.send_json(project_list())
            m = re.fullmatch(r"/api/projects/([\w\-]+)", path)
            if m:
                pid = m.group(1)
                man = P.load_manifest(pid)
                j = JOBS.get(pid)
                man["job"] = j.state(len(j.log)) if j else None
                man["summary"] = P.dataset_summary(man)
                man["training"] = training_for(man)
                return self.send_json(man)
            m = re.fullmatch(r"/api/jobs/([\w\-]+)", path)
            if m:
                j = JOBS.get(m.group(1))
                return self.send_json(j.state(int(q.get("since", 0))) if j else dict(status="none"))
            m = re.fullmatch(r"/files/([\w\-]+)/(.+)", path)
            if m:
                return self.send_file(safe_path(m.group(1), m.group(2)), q.get("dl") == "1")
            m = re.fullmatch(r"/api/peaks/([\w\-]+)/(.+)", path)
            if m:
                return self.send_json(A.peaks_cached(safe_path(m.group(1), m.group(2)), int(q.get("n", 800))))
            m = re.fullmatch(r"/api/text/([\w\-]+)", path)
            if m:
                pid = m.group(1)
                own = os.path.exists(P.text_path(pid))
                return self.send_json(dict(text=P.project_text(pid), own=own,
                                           name="свой текст" if own else os.path.basename(P.DEFAULT_TEXT)))
            m = re.fullmatch(r"/api/export/([\w\-]+)", path)
            if m:
                parts = [p for p in q.get("parts", "clean,tts,aug").split(",") if p in ("clean", "tts", "aug")]
                z = build_export(m.group(1), parts, q.get("onewav") == "1")
                try:
                    return self.send_file(z, True)
                finally:
                    try:
                        os.remove(z)
                    except OSError:
                        pass
            return self.err(404, "нет такого адреса")
        except FileNotFoundError as e:
            return self.err(404, str(e))
        except PermissionError:
            return self.err(403, "forbidden")
        except Exception as e:
            traceback.print_exc()
            return self.err(500, f"{type(e).__name__}: {e}")

    def do_PUT(self):
        return self.do_POST()

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        path = unquote(u.path)
        try:
            if path == "/api/upload":
                name = os.path.basename(unquote(q.get("name", "track.wav")))
                if not name.lower().endswith(A.AUDIO_EXT):
                    return self.err(400, "нужен аудиофайл: " + ", ".join(A.AUDIO_EXT))
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0:
                    return self.err(400, "пустой файл")

                def write(dst):
                    left = n
                    with open(dst, "wb") as f:
                        while left > 0:
                            b = self.rfile.read(min(1 << 20, left))
                            if not b:
                                raise IOError("загрузка оборвалась")
                            f.write(b)
                            left -= len(b)
                return self.send_json(dict(id=P.create_project(name, write)))
            if path == "/api/import":
                src = (self.body_json().get("path") or "").strip().strip('"')
                if not os.path.isfile(src):
                    return self.err(404, f"файл не найден: {src}")
                return self.send_json(dict(id=P.create_project(os.path.basename(src),
                                                              lambda dst: shutil.copyfile(src, dst))))
            if path == "/api/accent":     # stress preview for any text
                from studio import textprep as TP
                b = self.body_json()
                stressed, words = TP.accentuate(b.get("text", ""), accentizer(), b.get("dict") or {})
                return self.send_json(dict(stressed=stressed, words=words))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/(\w+)", path)
            if m:
                return self.project_post(m.group(1), m.group(2), q)
            m = re.fullmatch(r"/api/models/(\w+)/download", path)
            if m:
                key = m.group(1)
                if key not in M.ALL:
                    return self.err(404, "нет такой модели")
                if DOWNLOADS.get(key, {}).get("status") == "running":
                    return self.send_json(dict(ok=True))
                st = DOWNLOADS[key] = dict(status="running", done=0, total=0, file="")

                def go():
                    try:
                        M.ensure(key, lambda d, t, fn: st.update(done=d, total=t, file=fn), lambda s: None)
                        st["status"] = "done"
                    except Exception as e:
                        st.update(status="error", error=str(e))
                threading.Thread(target=go, daemon=True).start()
                return self.send_json(dict(ok=True))
            return self.err(404, "нет такого адреса")
        except FileNotFoundError as e:
            return self.err(404, str(e))
        except Exception as e:
            traceback.print_exc()
            return self.err(500, f"{type(e).__name__}: {e}")

    # ---- per project actions
    def project_post(self, pid, action, q):
        b = self.body_json() if action != "textfile" else {}
        if action == "run":
            if busy(pid):
                return self.err(409, "уже выполняется")
            job = P.Job(pid, b)
            JOBS[pid] = job
            man = P.load_manifest(pid)
            man["status"] = "queued"
            P.save_manifest(pid, man)
            QUEUE.put(job)
            return self.send_json(dict(ok=True, queued=QUEUE.qsize()))
        if action == "cancel":
            j = JOBS.get(pid)
            if j:
                j.cancel()
            return self.send_json(dict(ok=True))
        if action == "delete":
            if busy(pid):
                return self.err(409, "сначала отмените обработку")
            shutil.rmtree(P.project_dir(pid), ignore_errors=True)
            JOBS.pop(pid, None)
            return self.send_json(dict(ok=True))
        if action == "open":
            d = P.project_dir(pid)
            if b.get("sub"):
                d = safe_path(pid, b["sub"])
                os.makedirs(d, exist_ok=True) if not os.path.splitext(d)[1] else None
                if os.path.isfile(d):
                    d = os.path.dirname(d)
            open_in_explorer(d)
            return self.send_json(dict(ok=True))
        if action == "folder":       # build export/dataset and open it
            parts = [p for p in (b.get("parts") or ["clean", "tts", "aug"]) if p in ("clean", "tts", "aug")]
            d = build_folder(pid, parts)
            if not b.get("no_open"):
                try:
                    open_in_explorer(d)
                except Exception:
                    pass
            return self.send_json(dict(ok=True, path=d))
        if action == "text":         # save the project's own text ('' -> back to the default text)
            p = P.text_path(pid)
            t = b.get("text", "")
            if t.strip():
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w", encoding="utf-8").write(t)
            elif os.path.exists(p):
                os.remove(p)
            return self.send_json(dict(ok=True))
        # ---- edits of the manifest (not while a job runs: it would overwrite them)
        if busy(pid):
            return self.err(409, "дождитесь окончания обработки")
        man = P.load_manifest(pid)
        if action == "slice":        # include / exclude a slice of variant 1 or 2
            kind, f, inc = b["kind"], b["file"], bool(b["include"])
            lst = (man.get("clean") or {}).get("slices", []) if kind == "clean" else \
                (((man.get("tts") or {}).get("dataset") or {}).get("slices", []))
            for s in lst:
                if s["file"] == f:
                    s["include"] = inc
            man["summary"] = P.dataset_summary(man)
            P.save_manifest(pid, man)
            return self.send_json(dict(ok=True, summary=man["summary"]))
        if action == "block":        # TTS block: include / accept by ear / choose take / edit stress
            bl = man["tts"]["blocks"][int(b["i"])]
            for k in ("include", "accepted"):
                if k in b:
                    bl[k] = b[k]
            if "chosen" in b:
                bl["chosen"] = int(b["chosen"])
            if "edited" in b:
                bl["edited"] = b["edited"]
            P.save_manifest(pid, man)
            rebuild = any(k in b for k in ("include", "accepted", "chosen"))
            if rebuild:
                job = P.Job(pid, man.get("options") or {}, kind="rebuild")
                JOBS[pid] = job
                QUEUE.put(job)
            return self.send_json(dict(ok=True, rebuild=rebuild))
        if action == "dict":         # user stress dictionary: {word: '+'-marked form or ''}
            d = man.get("user_dict") or {}
            for w, v in (b.get("set") or {}).items():
                if v:
                    d[w.lower()] = v
                else:
                    d.pop(w.lower(), None)
            man["user_dict"] = d
            P.save_manifest(pid, man)
            return self.send_json(dict(ok=True, user_dict=d))
        if action == "redo":         # re-synthesise TTS blocks (after stress edits or a failed check)
            blocks = [int(i) for i in b.get("blocks", [])]
            if not blocks:
                return self.err(400, "не выбраны блоки")
            job = P.Job(pid, P.deep_merge(man.get("options") or {}, b.get("options") or {}), kind="tts_redo",
                        blocks=blocks)
            JOBS[pid] = job
            QUEUE.put(job)
            return self.send_json(dict(ok=True))
        if action == "rebuild":
            job = P.Job(pid, man.get("options") or {}, kind="rebuild")
            JOBS[pid] = job
            QUEUE.put(job)
            return self.send_json(dict(ok=True))
        if action == "rename":
            man["name"] = (b.get("name") or man["name"]).strip()[:120]
            P.save_manifest(pid, man)
            return self.send_json(dict(ok=True))
        return self.err(404, "нет такого действия")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    info = system_info()
    port = a.port
    for _ in range(20):
        try:
            srv = ThreadingHTTPServer((a.host, port), H)
            break
        except OSError:
            port += 1
    else:
        sys.exit("не нашёл свободный порт")
    srv.daemon_threads = True
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{port}/"
    print("=" * 64)
    print(f"  RVC STUDIO {VERSION}  ->  {url}")
    print(f"  устройство: {info['label']}")
    print(f"  проекты:    {P.PROJECTS}")
    print("  Закройте это окно, чтобы выключить студию.")
    print("=" * 64, flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

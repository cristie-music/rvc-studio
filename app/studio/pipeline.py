"""RVC Studio pipeline: track -> vocal -> lead only -> no echo/reverb -> datasets.

Stages (each cached in the project folder, re-run only when its inputs/options change):
  separate   vocals from the mix (Mel-Band RoFormer becruily)                         -> vocal/vocals.wav
  lead       lead voice without backing vocals (karaoke model)                         -> vocal/lead.wav
  dry        delay/echo + reverb removed (Sucial dereverb-echo v2, anvuew dereverb)   -> vocal/dry.wav
  clean      variant 1: clean voice, pauses/noise removed, sliced, scored              -> dataset/clean/*.wav
  aug        variant 3: formant-preserving pitch-shifted copies of the clean slices    -> dataset/aug/*.wav
  tts        variant 2: the singer's voice cloned by Qwen3-TTS reads a text (stress marks + back-check)
                                                                                      -> dataset/tts/*.wav
"""
import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback

import numpy as np
import soundfile as sf

from . import audio_io as A
from . import dataset as DS
from . import models as M

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP)
PROJECTS = os.environ.get("RVC_PROJECTS") or os.path.join(ROOT, "projects")
DEFAULT_TEXT = os.path.join(ROOT, "harry.txt")
SR = A.SR

DEFAULT_OPTIONS = dict(
    device="auto", quality="normal",                 # quality: high = overlap 4, normal = 2, fast = 1.5
    lead_only=True, dereverb="normal",               # dereverb: off | light | normal | strong
    variants=dict(clean=True, tts=True, aug=False),
    clean=dict(denoise="light", breaths="soften", min_len=2.0, max_len=10.0, target_sr=40000, level_db=-20,
               min_score=45),
    aug=dict(semis=[-2, -1, 1, 2]),
    tts=dict(model="tts", max_blocks=60, takes=3, strictness="normal", temperature=0.8, clone="xvec",
             ref_file=None, ref_text=None, seed=1, batch=0),
)


def deep_merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def new_id(name):
    base = re.sub(r"[^\w\-]+", "_", os.path.splitext(name)[0], flags=re.U).strip("_")[:40] or "track"
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{base}"


def project_dir(pid):
    if not re.fullmatch(r"[\w\-]+", pid, flags=re.U):
        raise ValueError("bad project id")
    return os.path.join(PROJECTS, pid)


def load_manifest(pid):
    with open(os.path.join(project_dir(pid), "project.json"), encoding="utf-8") as f:
        return json.load(f)


_SAVE_LOCK = threading.Lock()


def save_manifest(pid, man):
    p = os.path.join(project_dir(pid), "project.json")
    with _SAVE_LOCK:
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)


def create_project(src_name, write_fn):
    pid = new_id(src_name)
    d = project_dir(pid)
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(src_name)[1].lower() or ".wav"
    dst = os.path.join(d, "source" + ext)
    write_fn(dst)
    info = sf.info(dst) if ext in (".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3") else None
    man = dict(id=pid, name=os.path.splitext(os.path.basename(src_name))[0], source="source" + ext,
               source_name=os.path.basename(src_name), created=time.strftime("%Y-%m-%d %H:%M:%S"),
               duration=round(info.duration, 2) if info else None, status="new", options=None,
               vocal={}, clean=None, aug=None, tts=None, text=None, user_dict={})
    save_manifest(pid, man)
    return pid


def text_path(pid):
    return os.path.join(project_dir(pid), "tts", "text.txt")


def project_text(pid):
    p = text_path(pid)
    if os.path.exists(p):
        return open(p, encoding="utf-8-sig").read()
    if os.path.exists(DEFAULT_TEXT):
        return open(DEFAULT_TEXT, encoding="utf-8-sig").read()
    return ""


class Cancelled(Exception):
    pass


def _sig(*parts):
    return hashlib.md5(json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


class Job:
    """One run. kind: process (stages from options) | tts_redo (blocks) | rebuild (datasets from saved state)."""

    def __init__(self, pid, options, kind="process", blocks=None):
        self.pid, self.kind, self.blocks = pid, kind, blocks or []
        self.options = deep_merge(DEFAULT_OPTIONS, options or {})
        self.progress, self.step = 0.0, "В очереди"
        self.steps, self.log = [], []
        self.status, self.error = "queued", None
        self.started = self.finished = None
        self._cancel = False
        self._lock = threading.Lock()

    # ---- state helpers
    def say(self, msg):
        with self._lock:
            self.log.extend(str(msg).split("\n"))
            if len(self.log) > 5000:
                del self.log[:1000]
        print(msg, flush=True)

    def set_step(self, sid, state, note=None):
        for s in self.steps:
            if s["id"] == sid:
                s["state"] = state
                if note is not None:
                    s["note"] = note

    def cancel(self):
        self._cancel = True

    def check(self):
        if self._cancel:
            raise Cancelled()

    def state(self, since=0):
        with self._lock:
            return dict(pid=self.pid, kind=self.kind, status=self.status, progress=round(self.progress, 4),
                        step=self.step, steps=self.steps, error=self.error, log=self.log[since:],
                        log_len=len(self.log), started=self.started, finished=self.finished)

    def prog(self, p0, p1, title):
        def f(frac, msg=""):
            self.progress = p0 + (p1 - p0) * max(0.0, min(1.0, frac))
            self.step = f"{title}: {msg}" if msg else title
            self.check()
        return f

    # ---- run
    def run(self):
        self.status, self.started = "running", time.strftime("%Y-%m-%d %H:%M:%S")
        man = load_manifest(self.pid)
        man["status"] = "running"
        if self.kind == "process":
            man["options"] = self.options
        save_manifest(self.pid, man)
        try:
            if self.kind == "process":
                self._process(man)
            elif self.kind == "tts_redo":
                self._tts_redo(man)
            elif self.kind == "rebuild":
                self._rebuild(man)
            self.status = man["status"] = "done"
            self.progress, self.step = 1.0, "Готово"
        except Cancelled:
            self.status = man["status"] = "cancelled"
            self.step = "Отменено"
            self.say("\n! Отменено пользователем")
        except Exception as e:
            traceback.print_exc()
            self.status = man["status"] = "error"
            self.error = man["error"] = f"{type(e).__name__}: {e}"
            self.step = "Ошибка"
            self.say("\n! ОШИБКА: " + self.error)
            self.say(traceback.format_exc())
        finally:
            try:
                from . import meter2, verify
                verify.free_asr()
                meter2.free()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self.finished = man["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        man["log_tail"] = self.log[-300:]
        save_manifest(self.pid, man)

    # ================================================================ process
    def _process(self, man):
        o = self.options
        v = o["variants"]
        plan = [("separate", "Вокал из трека"), ("lead", "Основной голос без бэков"), ("dry", "Удаление эха и ревера")]
        if v.get("clean") or v.get("aug") or v.get("tts"):
            plan.append(("clean", "Вариант 1: чистый голос → нарезка"))
        if v.get("aug"):
            plan.append(("aug", "Вариант 3: копии со сдвигом высоты"))
        if v.get("tts"):
            plan.append(("tts", "Вариант 2: озвучка текста голосом певца"))
        plan.append(("pack", "Сборка датасета"))
        pdir = project_dir(self.pid)
        if not v.get("aug") and man.get("aug"):          # a switched-off variant leaves the dataset
            man["aug"] = None
            shutil.rmtree(os.path.join(pdir, "dataset", "aug"), ignore_errors=True)
        if not v.get("tts") and (man.get("tts") or {}).get("dataset"):
            man["tts"]["dataset"] = None                  # takes stay in tts/ for later, only the slices go
            shutil.rmtree(os.path.join(pdir, "dataset", "tts"), ignore_errors=True)
        self.steps = [dict(id=i, title=t, state="wait", note="") for i, t in plan]
        weights = dict(separate=0.15, lead=0.1, dry=0.15, clean=0.1, aug=0.05, tts=0.4, pack=0.05)
        total = sum(weights[i] for i, _ in plan)
        p = 0.0
        for sid, title in plan:
            p1 = p + weights[sid] / total
            self.set_step(sid, "run")
            self.say(f"\n=== {title} ===")
            t0 = time.time()
            note = getattr(self, "_st_" + sid)(man, self.prog(p, p1, title)) or ""
            self.set_step(sid, "done", (note + f" · {time.time() - t0:.0f} с").strip(" ·"))
            save_manifest(self.pid, man)
            p = p1

    # ---- vocal chain
    def _vocal_path(self, name):
        return os.path.join(project_dir(self.pid), "vocal", name)

    def _separate_cached(self, man, stage, key, inp, outs, prog, extra=None):
        """run separation model `key` on audio file `inp` -> files {stem: name}; cached by signature."""
        o = self.options
        sig = _sig(stage, key, os.path.getmtime(inp), o["quality"], extra)
        st = man["vocal"].get(stage) or {}
        if st.get("sig") == sig and all(os.path.exists(self._vocal_path(f)) for f in outs.values()):
            self.say("  из кэша")
            prog(1.0, "из кэша")
            return st
        M.ensure(key, lambda d, t, fn: prog(0.0, f"загрузка модели {d / 2**20:.0f}/{t / 2**20:.0f} MB"), self.say)
        from . import separator as S
        x, _ = A.load_stereo(inp)
        overlap = dict(high=4, normal=2, fast=1.5)[o["quality"]]
        stems, info = S.separate(x.T, key, o["device"], overlap, prog, self.say)
        self.say(f"  {info['model']} · {info['device']} · {info['seconds']} с")
        files = {}
        for stem, name in outs.items():
            y = stems[stem].T
            A.write_wav(self._vocal_path(name), y, SR, "FLOAT")
            A.write_mp3(self._vocal_path(name[:-4] + ".mp3"), y, SR)
            files[stem] = name
        st = dict(sig=sig, files=files, model=info["model"], device=info["device"], seconds=info["seconds"])
        man["vocal"][stage] = st
        return st

    def _st_separate(self, man, prog):
        pdir = project_dir(self.pid)
        src = os.path.join(pdir, man["source"])
        if not os.path.exists(self._vocal_path("mix.mp3")):
            x, _ = A.load_stereo(src)
            A.write_mp3(self._vocal_path("mix.mp3"), x, SR)
            man["duration"] = round(len(x) / SR, 2)
        st = self._separate_cached(man, "separate", "vocals", src,
                                   {"vocals": "vocals.wav", "rest": "instrumental.wav"}, prog)
        return st["model"].split("·")[-1].strip()

    def _st_lead(self, man, prog):
        if not self.options["lead_only"]:
            man["vocal"]["lead"] = None
            prog(1.0)
            return "выключено"
        st = self._separate_cached(man, "lead", "karaoke", self._vocal_path("vocals.wav"),
                                   {"vocals": "lead.wav", "instrumental": "backing.wav"}, prog)
        lead, _ = A.load_stereo(self._vocal_path("lead.wav"))
        back, _ = A.load_stereo(self._vocal_path("backing.wav"))
        st["backing_db"] = round(A.rms_db(back) - A.rms_db(lead), 1)
        self.say(f"  бэк-вокал тише основного на {-st['backing_db']:.1f} дБ")
        return f"бэки {st['backing_db']:+.1f} дБ к голосу"

    def _st_dry(self, man, prog):
        mode = self.options["dereverb"]
        src = self._vocal_path("lead.wav" if self.options["lead_only"] else "vocals.wav")
        chain = dict(off=[], light=["dereverb"], normal=["dereverb_echo"], strong=["dereverb_echo", "dereverb"])[mode]
        cur, notes = src, []
        for k, key in enumerate(chain):
            def sub(frac, msg="", k=k):
                prog((k + frac) / len(chain), msg)
            name = f"dry_{key}.wav"
            target = "dry" if key == "dereverb_echo" else "noreverb"
            self._separate_cached(man, f"dry_{key}", key, cur, {target: name, "rest": f"removed_{key}.wav"}, sub,
                                  extra=[cur, mode])
            cur = self._vocal_path(name)
            notes.append("эхо+ревер" if key == "dereverb_echo" else "ревер")
        x, _ = A.load_stereo(cur)
        A.write_wav(self._vocal_path("dry.wav"), x, SR, "FLOAT")
        A.write_mp3(self._vocal_path("dry.mp3"), x, SR)
        if chain:   # everything that was removed, for listening
            y, _ = A.load_stereo(src)
            A.write_mp3(self._vocal_path("removed_fx.mp3"), y - x, SR)
        man["vocal"]["dry"] = dict(mode=mode, chain=chain, file="dry.wav")
        prog(1.0)
        return ", ".join(notes) or "без обработки"

    # ---- variant 1
    def _st_clean(self, man, prog):
        o = self.options["clean"]
        pdir = project_dir(self.pid)
        x, _ = A.load_stereo(self._vocal_path("dry.wav"))
        out = os.path.join(pdir, "dataset", "clean")
        r = DS.build(x, SR, out, "voice", o, self.say, prog)
        man["clean"] = dict(options=o, **r)
        s = r["stats"]
        self.say(f"  голос звучит {s['voice_s']:.0f} с из {s['source_s']:.0f} с; кусков {s['slices']}, "
                 f"в датасет {s['included']} ({s['dataset_s']:.0f} с)")
        return f"{s['included']}/{s['slices']} кусков, {s['dataset_s'] / 60:.1f} мин"

    # ---- variant 3
    def _st_aug(self, man, prog):
        pdir = project_dir(self.pid)
        semis = [float(v) for v in self.options["aug"]["semis"] if float(v) != 0]
        res = DS.augment(os.path.join(pdir, "dataset", "clean"), man["clean"]["slices"],
                         os.path.join(pdir, "dataset", "aug"), "aug", semis, man["clean"]["stats"]["target_sr"],
                         self.say, prog)
        man["aug"] = dict(semis=semis, files=res, dataset_s=round(sum(r["dur"] for r in res), 1))
        return f"{len(res)} копий, {man['aug']['dataset_s'] / 60:.1f} мин"

    # ---- variant 2
    def _pick_reference(self, man):
        """best 5..12 s piece of clean voice: high score, much voiced singing; returns (file, slice)."""
        to = self.options["tts"]
        sl = [s for s in (man.get("clean") or {}).get("slices", []) if s.get("include")]
        if to.get("ref_file"):
            for s in sl + (man.get("clean") or {}).get("slices", []):
                if s["file"] == to["ref_file"]:
                    return s
        if not sl:
            raise RuntimeError("нет чистых кусков голоса для образца (сначала вариант 1)")
        def rank(s):
            d = s["dur"]
            fit = 1.0 if 5 <= d <= 12 else 0.6 if 3 <= d <= 15 else 0.2
            return s["score"] * fit + 30 * s.get("voiced", 0) - 200 * s.get("flatness", 0)
        return max(sl, key=rank)

    def _tts_prepare(self, man, prog):
        """reference + text blocks (with stress) -> man['tts'] skeleton; keeps finished takes of unchanged blocks."""
        from . import textprep as TP
        to = self.options["tts"]
        pdir = project_dir(self.pid)
        tdir = os.path.join(pdir, "tts")
        os.makedirs(os.path.join(tdir, "takes"), exist_ok=True)
        ref = self._pick_reference(man)
        y, sr = sf.read(os.path.join(pdir, "dataset", "clean", ref["file"]), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        y = y[: int(sr * 15)]
        sf.write(os.path.join(tdir, "ref.wav"), y, sr)
        prog(0.02, "образец голоса: " + ref["file"])
        old = man.get("tts") or {}
        ref_text = to.get("ref_text") or (old.get("ref", {}).get("text") if old.get("ref", {}).get("file") == ref["file"]
                                          else None)
        if not ref_text and to.get("clone", "xvec") == "icl":
            from . import verify as V
            asr = V.load_asr(self.options["device"], self.say)
            ref_text = " ".join(w["w"] for w in V.transcribe(asr, y, sr)).strip()
            self.say(f"  образец {ref['file']} ({ref['dur']} с), в нём слышно: «{ref_text}»")
        acc = TP.load_accentizer(M.repo_dir("ruaccent"), self.say) if M.installed("ruaccent") or self._ensure_ruaccent() else None
        text = project_text(self.pid)
        blocks_txt = TP.chunks(text)
        if to.get("max_blocks"):
            blocks_txt = blocks_txt[: int(to["max_blocks"])]
        old_blocks = {b["text"]: b for b in old.get("blocks", [])}
        blocks = []
        for i, t in enumerate(blocks_txt):
            stressed, words = TP.accentuate(t, acc, man.get("user_dict"))
            ob = old_blocks.get(t)
            b = dict(i=i, text=t, stressed=stressed, words=words, takes=[], chosen=None, status="new")
            same_voice = old.get("clone") == to.get("clone", "xvec") and old.get("model") == to["model"]
            if ob and same_voice and ob.get("stressed") == stressed and ob.get("ref") == ref["file"]:
                b.update(takes=ob.get("takes", []), chosen=ob.get("chosen"), status=ob.get("status", "new"),
                         accepted=ob.get("accepted"))
            b["ref"] = ref["file"]
            blocks.append(b)
        man["tts"] = dict(ref=dict(file=ref["file"], text=ref_text, dur=ref["dur"]), blocks=blocks,
                          model=to["model"], clone=to.get("clone", "xvec"),
                          text_name=os.path.basename(text_path(self.pid)) if os.path.exists(text_path(self.pid))
                          else os.path.basename(DEFAULT_TEXT))
        save_manifest(self.pid, man)
        return y, sr, ref_text

    def _ensure_ruaccent(self):
        M.ensure("ruaccent", log=self.say)
        return True

    def _synth(self, man, todo, prog, fresh_seed=False):
        """generate takes for blocks `todo` until one passes the check or `takes` attempts are used."""
        from . import textprep as TP
        from . import tts as TT
        from . import verify as V
        to = self.options["tts"]
        pdir = project_dir(self.pid)
        tdir = os.path.join(pdir, "tts")
        ref_y, ref_sr = sf.read(os.path.join(tdir, "ref.wav"), dtype="float32")
        ref_text = (man["tts"]["ref"].get("text") or "") if man["tts"].get("clone") == "icl" else ""
        prog(0.03, "загрузка модели озвучки")
        M.ensure(to["model"], lambda d, t, fn: prog(0.03, f"загрузка модели {d / 2**20:.0f}/{t / 2**20:.0f} MB"),
                 self.say)
        tts = TT.TTS(to["model"], self.options["device"], self.say)
        tts.set_voice(ref_y, ref_sr, ref_text)
        ref_emb = tts.embed(ref_y, ref_sr)
        asr = V.load_asr(self.options["device"], self.say)
        blocks = man["tts"]["blocks"]
        gpu = tts.dinfo["kind"] != "cpu"
        batch = int(to.get("batch") or (4 if gpu and (tts.dinfo.get("vram_gb") or 0) >= 12 else 2 if gpu else 1))
        max_takes = max(1, int(to["takes"]))
        t0, done = time.time(), 0
        queue = list(todo)
        attempt = {i: len(blocks[i]["takes"]) for i in queue}
        while queue:
            self.check()
            grp = queue[:batch]
            queue = queue[batch:]
            # the ru-stress model follows U+0301 marks; a plain Base model would read them as sounds -> plain text
            marks = M.ALL[to["model"]].get("marks", False)
            texts = [TP.plus_to_acute(blocks[i]["stressed"]) if marks else TP.strip_marks(blocks[i]["stressed"])
                     for i in grp]
            seed = int(to.get("seed", 1)) * 1000 + min(attempt[i] for i in grp) + (int(time.time()) % 997 if fresh_seed else 0)
            try:
                wavs, sr = tts.say(texts, seed=seed, temperature=float(to["temperature"]))
            except Exception as e:
                if "out of memory" in str(e).lower() and batch > 1:
                    import torch
                    torch.cuda.empty_cache()
                    batch = max(1, batch // 2)
                    self.say(f"  ! мало видеопамяти -> по {batch} блока")
                    queue = grp + queue
                    continue
                raise
            for i, w in zip(grp, wavs):
                b = blocks[i]
                attempt[i] += 1
                w = trim_silence(w, sr)
                name = f"b{i + 1:04d}_t{attempt[i]}.wav"
                sf.write(os.path.join(tdir, "takes", name), w, sr, subtype="PCM_16")
                chk = V.check(w, sr, b["stressed"], b["words"], asr, to["strictness"], self.options["device"],
                              self.say)
                sim = float(ref_emb @ tts.embed(w, sr))       # same voice as the reference? (cosine, 1 = same)
                chk["score"] = round(chk["score"] + max(0.0, 0.7 - sim) * 4, 3)
                take = dict(file=name, seed=seed, dur=round(len(w) / sr, 2), sim=round(sim, 3), **chk)
                b["takes"].append(take)
                best = min(range(len(b["takes"])), key=lambda k: b["takes"][k]["score"])
                b["chosen"] = best
                b["status"] = "ok" if b["takes"][best]["ok"] else "check"
                if b.get("accepted") is None:
                    b["include"] = True
                mark = "✓" if chk["ok"] else "✗"
                why = "; ".join(chk["issues"] + ([f"WER {chk['wer']:.0%}"] if not chk["passed_words"] else []) +
                                ([f"ударения? {', '.join(TP.set_stress(s['w'], s['detected']) for s in chk['suspicious'][:3])}"]
                                 if chk["suspicious"] else []))
                self.say(f"  [{i + 1}] дубль {attempt[i]} {mark} WER {chk['wer']:.0%} голос {sim:.2f} "
                         f"ударений {chk['stress_checked']} ({take['dur']} с) {why}")
                if not chk["ok"] and attempt[i] < max_takes:
                    queue.append(i)
                else:
                    done += 1
            el = time.time() - t0
            prog(0.05 + 0.9 * done / max(1, len(todo)),
                 f"блок {done}/{len(todo)}, осталось ~{el / max(1, done) * (len(todo) - done) / 60:.0f} мин")
            save_manifest(self.pid, man)
        tts.close()

    def _st_tts(self, man, prog):
        self._tts_prepare(man, prog)
        blocks = man["tts"]["blocks"]
        todo = [b["i"] for b in blocks if not b["takes"]]
        self.say(f"  блоков текста: {len(blocks)}, озвучить: {len(todo)}"
                 + (f" (готовых: {len(blocks) - len(todo)})" if len(todo) < len(blocks) else ""))
        if todo:
            self._synth(man, todo, prog)
        ok = sum(b["status"] == "ok" for b in blocks)
        return f"{ok}/{len(blocks)} блоков прошли проверку"

    def _tts_redo(self, man):
        self.steps = [dict(id="tts", title="Переозвучка блоков", state="run", note=""),
                      dict(id="pack", title="Сборка датасета", state="wait", note="")]
        prog = self.prog(0, 0.9, "Переозвучка")
        from . import textprep as TP
        acc = TP.load_accentizer(M.repo_dir("ruaccent"), self.say) if self._ensure_ruaccent() else None
        blocks = man["tts"]["blocks"]
        for i in self.blocks:   # stress may have been edited: re-accentuate, keep manual marks
            b = blocks[i]
            src = b.get("edited") or b["text"]
            b["stressed"], b["words"] = TP.accentuate(src, acc, man.get("user_dict"))
            b["takes"], b["chosen"], b["status"] = [], None, "new"
        save_manifest(self.pid, man)
        self._synth(man, list(self.blocks), prog, fresh_seed=True)
        self.set_step("tts", "done")
        self.set_step("pack", "run")
        self._st_pack(man, self.prog(0.9, 1.0, "Сборка"))
        self.set_step("pack", "done")

    def _rebuild(self, man):
        self.steps = [dict(id="pack", title="Сборка датасета", state="run", note="")]
        self._st_pack(man, self.prog(0, 1.0, "Сборка"))
        self.set_step("pack", "done")

    # ---- pack: TTS takes -> slices, summary
    def _st_pack(self, man, prog):
        pdir = project_dir(self.pid)
        tsr = (man.get("clean") or {}).get("stats", {}).get("target_sr") or self.options["clean"]["target_sr"]
        notes = []
        if man.get("tts") and man["tts"].get("blocks"):
            tdir = os.path.join(pdir, "tts", "takes")
            parts, gap = [], None
            for b in man["tts"]["blocks"]:
                if b.get("chosen") is None or not b.get("include", True):
                    continue
                if b["status"] != "ok" and not b.get("accepted"):
                    continue          # failed the check and not accepted by ear: keep out of the dataset
                y, sr = sf.read(os.path.join(tdir, b["takes"][b["chosen"]]["file"]), dtype="float32")
                gap = np.zeros(int(sr * 0.6), np.float32)
                parts += [y, gap]
            out = os.path.join(pdir, "dataset", "tts")
            if parts:
                y = np.concatenate(parts)
                o = dict(self.options["clean"], denoise="off", breaths="keep", min_score=30, target_sr=tsr,
                         max_len=self.options["clean"]["max_len"])
                r = DS.build(y, sr, out, "tts", o, self.say, prog)
                man["tts"]["dataset"] = r
                notes.append(f"озвучка {r['stats']['dataset_s'] / 60:.1f} мин")
            else:
                shutil.rmtree(out, ignore_errors=True)
                man["tts"]["dataset"] = None
        prog(1.0)
        tot = dataset_summary(man)
        man["summary"] = tot
        self.say(f"  итого в датасете: {tot['total_s'] / 60:.1f} мин "
                 f"(голос {tot['clean_s'] / 60:.1f}, озвучка {tot['tts_s'] / 60:.1f}, копии {tot['aug_s'] / 60:.1f})")
        try:
            from . import device as D
            d = D.detect(self.options["device"])
            t = recommend_training(tot, d.get("vram_gb"), d["kind"])
            if t:
                self.say(f"  обучение RVC: {t['epochs']} эпох ({t['epochs_low']}-{t['epochs_high']}), batch {t['batch']}, "
                         f"~{t['total_steps']} шагов, сохранять каждые {t['save_every']}")
        except Exception:
            pass
        return f"{tot['total_s'] / 60:.1f} мин всего" + (", " + ", ".join(notes) if notes else "")


def trim_silence(y, sr, thr_db=-45, pad=0.08):
    if len(y) == 0:
        return y
    db = DS.frame_db(y, sr)
    act = np.where(db > max(db.max() - 45, thr_db))[0]
    if len(act) == 0:
        return y
    a = max(0, int((act[0] * DS.FR - pad) * sr))
    b = min(len(y), int((act[-1] * DS.FR + DS.WIN + pad) * sr))
    z = y[a:b].copy()
    f = min(int(0.01 * sr), len(z) // 4)
    if f:
        z[:f] *= np.linspace(0, 1, f)
        z[-f:] *= np.linspace(1, 0, f)
    return z


EPOCH_TABLE = (   # effective minutes of DIFFERENT material -> (recommended, low, high) epochs for RVC v2 / Applio
    (3, (400, 300, 500)), (6, (300, 250, 400)), (12, (250, 200, 300)), (25, (200, 150, 250)),
    (45, (150, 100, 200)), (1e9, (100, 60, 150)))


def recommend_training(summary, vram_gb=None, kind="cuda"):
    """Epochs / batch / steps for RVC training of this dataset.
    Pitch-shifted copies are near-duplicates of the clean voice: they add steps per epoch but little new material,
    so they count 30 % towards the 'effective' length used for the epoch table."""
    total = summary.get("total_s") or 0
    if total <= 0:
        return None
    eff_min = (summary.get("clean_s", 0) + summary.get("tts_s", 0) + 0.3 * summary.get("aug_s", 0)) / 60
    rec, lo, hi = next(v for lim, v in EPOCH_TABLE if eff_min < lim)
    if kind == "cpu" or not vram_gb:
        batch = 4
    else:
        batch = 4 if vram_gb < 8 else 6 if vram_gb < 12 else 8 if vram_gb < 20 else 16
    segments = max(1, int(total / 3.5))            # RVC re-cuts audio into ~3.7 s pieces
    spe = -(-segments // batch)
    notes = []
    if eff_min < 5:
        notes.append("данных мало (< 5 мин): модель может быть нестабильной; добавьте озвучку или другие треки")
    if summary.get("aug_s", 0) > summary.get("clean_s", 0) + summary.get("tts_s", 0):
        notes.append("копий со сдвигом высоты больше, чем настоящего голоса: лучше уменьшить их число")
    notes.append("следите за графиком потерь (TensorBoard) или включите Overtraining Detector в Applio: "
                 "если качество перестаёт расти, берите более раннюю сохранённую эпоху")
    return dict(epochs=rec, epochs_low=lo, epochs_high=hi, batch=batch, effective_min=round(eff_min, 1),
                segments=segments, steps_per_epoch=spe, total_steps=spe * rec, save_every=max(10, rec // 10),
                sample_rate=None, notes=notes)


def dataset_summary(man):
    c = (man.get("clean") or {}).get("slices", [])
    clean_s = sum(s["dur"] for s in c if s.get("include"))
    aug_s = (man.get("aug") or {}).get("dataset_s", 0) or 0
    tts_d = ((man.get("tts") or {}).get("dataset") or {})
    tts_s = sum(s["dur"] for s in tts_d.get("slices", []) if s.get("include"))
    return dict(clean_s=round(clean_s, 1), aug_s=round(aug_s, 1), tts_s=round(tts_s, 1),
                total_s=round(clean_s + aug_s + tts_s, 1),
                files=sum(1 for s in c if s.get("include")) + len((man.get("aug") or {}).get("files", [])) +
                sum(1 for s in tts_d.get("slices", []) if s.get("include")))

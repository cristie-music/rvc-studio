"""Back-check of every synthesised block (the idea of Speech Studio's audio/qc.rs, extended with a stress check).

1) Words: faster-whisper (large-v3-turbo) hears the take; written vs heard words are compared by word-level
   Levenshtein (WER). Thresholds as in Speech Studio: strict 10 % / normal 20 % / lenient 35 %, but never less than
   1 / 2 / 3 words for short blocks, never more than 60 %.
2) Timing: silence > 1 s at the ends, gaps > 1.5 s inside, a take much longer/shorter than the text needs,
   a 2-4 word run repeated in the take but not in the text (TTS stutter).
3) Stress (an addition of RVC Studio): meter2 (siriusfreak) aligns the text to the take with a Russian wav2vec2 CTC
   model and scores every vowel of every word; the stressed vowel it hears is compared with the '+' mark from
   RUAccent / the user. A confident disagreement (margin >= 0.5) = wrong stress -> the block is re-generated with
   another seed. Words the aligner cannot find in the audio (confidence < 0.5) = "probably not said as written".
"""
import difflib
import glob
import os
import re
import sys

import numpy as np

from . import models as M
from . import textprep as T

MAX_THRESHOLD = 0.6
_ASR = None


# ---------------------------------------------------------------- ASR
def _cuda_dlls():
    """ctranslate2 needs cuBLAS / cuDNN 9 for CUDA 12: PyTorch (cu126) ships exactly those DLLs - reuse them."""
    try:
        import torch
        d = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.name == "nt" and os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        for p in glob.glob(os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "nvidia", "*", "bin")):
            os.add_dll_directory(p)
    except Exception:
        pass


def load_asr(want="auto", log=print):
    global _ASR
    if _ASR is not None:
        return _ASR
    _cuda_dlls()
    from faster_whisper import WhisperModel
    M.ensure("asr", log=log)
    path = M.repo_dir("asr")
    tries = []
    if want != "cpu":
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                tries.append(("cuda", "float16"))
        except Exception:
            pass
    tries.append(("cpu", "int8"))
    err = None
    for dev, ct in tries:
        try:
            m = WhisperModel(path, device=dev, compute_type=ct, cpu_threads=max(1, (os.cpu_count() or 2) - 1))
            m.transcribe(np.zeros(16000, np.float32), language="ru")      # surfaces missing CUDA DLLs right now
            log(f"  распознавание речи: Whisper large-v3-turbo на {'GPU' if dev == 'cuda' else 'CPU'}")
            _ASR = m
            return m
        except Exception as e:
            err = e
            log(f"  ! распознавание на {dev} не запустилось ({str(e)[:120]})")
    raise RuntimeError(f"распознавание речи не запустилось: {err}")


def free_asr():
    global _ASR
    _ASR = None


def to16k(y, sr):
    from scipy.signal import resample_poly
    g = np.gcd(int(sr), 16000)
    return resample_poly(y, 16000 // g, int(sr) // g).astype(np.float32)


def transcribe(model, y, sr, prompt=None):
    segs, _ = model.transcribe(to16k(y, sr), language="ru", word_timestamps=True, vad_filter=False,
                               initial_prompt=prompt, beam_size=5, temperature=0.0, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in s.words or []:
            words.append(dict(w=w.word.strip(), s=round(w.start, 3), e=round(w.end, 3), p=round(w.probability, 2)))
    return words


# ---------------------------------------------------------------- words
def norm_word(w):
    w = T.strip_marks(w).lower().replace("ё", "е")
    return re.sub(r"[^\w]", "", w)


def num_words(tok):
    """'4' -> ['четыре'] (ASR writes numbers as digits, the text has them as words)."""
    if tok.isdigit():
        try:
            from num2words import num2words
            return [norm_word(x) for x in num2words(int(tok), lang="ru").split()]
        except Exception:
            pass
    return [tok]


def tokens(text):
    out = []
    for x in re.split(r"[\s\-—]+", text):
        t = norm_word(x)
        if t:
            out += num_words(t)
    return out


def same(a, b):
    """equal words, or near-equal long words (names: Дурсль / Дурсель, Граннингс / Гранингс)."""
    return a == b or (min(len(a), len(b)) >= 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75)


def align(exp, heard):
    """word-level Levenshtein -> (sub, ins, dele, pairs[(i_exp, j_heard)] of equal words)."""
    n, m = len(exp), len(heard)
    D_ = np.zeros((n + 1, m + 1), np.int32)
    D_[:, 0] = np.arange(n + 1)
    D_[0, :] = np.arange(m + 1)
    eq = [[same(a, b) for b in heard] for a in exp]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if eq[i - 1][j - 1] else 1
            D_[i, j] = min(D_[i - 1, j] + 1, D_[i, j - 1] + 1, D_[i - 1, j - 1] + c)
    i, j, sub, ins, dele, pairs = n, m, 0, 0, 0, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D_[i, j] == D_[i - 1, j - 1] + (not eq[i - 1][j - 1]):
            if eq[i - 1][j - 1]:
                pairs.append((i - 1, j - 1))
            else:
                sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and D_[i, j] == D_[i - 1, j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, ins, dele, pairs[::-1]


def threshold(words, strictness="normal"):
    base, slack = {"strict": (0.10, 1.0), "lenient": (0.35, 3.0)}.get(strictness, (0.20, 2.0))
    return min(max(base, slack / max(1, words)), MAX_THRESHOLD)


def repeated(exp, heard):
    """a 2-4 word run doubled in the take but not in the text."""
    def has_rep(ws, n):
        return {tuple(ws[i:i + n]) for i in range(len(ws) - 2 * n + 1) if ws[i:i + n] == ws[i + n:i + 2 * n]}
    for n in (4, 3, 2):
        r = has_rep(heard, n) - has_rep(exp, n)
        if r:
            return " ".join(next(iter(r)))
    return None


# ---------------------------------------------------------------- the check
_MARKED = re.compile(r"[А-Яа-яЁё+́]+(?:-[А-Яа-яЁё+́]+)*")


def stress_check(y, sr, stressed_text, want="auto", log=print, margin_min=0.5):
    """meter2 on the take -> (checked, suspicious[], misread[]). Expected vowel = the '+' mark (or ё)."""
    from . import meter2
    if not meter2.available():
        return 0, [], [], False
    marked = [m.group(0) for m in _MARKED.finditer(T.acute_to_plus(stressed_text))]
    res = meter2.judge(y, sr, stressed_text, want, log)
    checked, sus, misread = 0, [], []
    for mw, r in zip(marked, res):
        plain = T.strip_marks(mw)
        vp = meter2.vowel_pos(plain)
        if r.get("align") is not None and r["align"] < meter2.CONF_MIN:
            if len(plain) >= 3:      # 1-2 letter words (и, в, на) are often swallowed: not worth a flag
                misread.append(dict(w=plain, s=r["start"], e=r["end"], align=r["align"]))
            continue
        if len(vp) < 2 or "-" in plain or r.get("best") is None:
            continue
        exp = meter2.marked_index(mw)
        if exp is None and "ё" in plain.lower():
            exp = [plain.lower()[i] for i in vp].index("ё")
        if exp is None:
            continue
        checked += 1
        if r["best"] != exp and (r["margin"] or 0) >= margin_min:
            sus.append(dict(w=plain, expected=exp, detected=r["best"], conf=r["margin"], s=r["start"], e=r["end"],
                            scores=r["vowel_scores"]))
    return checked, sus, misread, True


def check(y, sr, stressed_text, words_info, asr_model, strictness="normal", want="auto", log=print):
    """y: mono take. stressed_text: block text with '+' marks.
    -> dict(ok, score, wer, heard, issues[], stress_checked, suspicious[], misread[])"""
    plain = T.strip_marks(stressed_text)
    exp = tokens(plain)
    raw = transcribe(asr_model, y, sr)
    heard, heard_w = [], []
    for w in raw:
        for part in re.split(r"[\-—]+", w["w"]):
            t = norm_word(part)
            for tok in (num_words(t) if t else []):
                heard.append(tok)
                heard_w.append(w)
    sub, ins, dele, pairs = align(exp, heard)
    wer = (sub + ins + dele) / max(1, len(exp)) if exp else (0.0 if not heard else 1.0)
    thr = threshold(len(exp), strictness)
    issues = []
    dur = len(y) / sr
    if raw:
        if raw[0]["s"] > 1.0:
            issues.append(f"тишина в начале {raw[0]['s']:.1f} с")
        if dur - raw[-1]["e"] > 1.5:
            issues.append(f"хвост {dur - raw[-1]['e']:.1f} с")
        for a, b in zip(raw[:-1], raw[1:]):
            if b["s"] - a["e"] > 1.5:
                issues.append(f"пауза {b['s'] - a['e']:.1f} с на {a['e']:.1f} с")
                break
    letters = len(re.sub(r"[^А-Яа-яЁё]", "", plain))
    need = 0.065 * letters
    if dur > need * 2.2 + 1.5:
        issues.append(f"слишком длинно ({dur:.1f} с при ожидаемых ~{need:.0f} с)")
    if dur < need * 0.45:
        issues.append(f"слишком коротко ({dur:.1f} с при ожидаемых ~{need:.0f} с)")
    rep = repeated(exp, heard)
    if rep:
        issues.append(f"повтор: «{rep}»")
    checked, sus, misread, have_meter = stress_check(y, sr, stressed_text, want, log)
    passed = wer <= thr
    stress_ok = len(sus) == 0 or (len(sus) == 1 and checked >= 12)
    ok = passed and not issues and stress_ok
    score = wer * 10 + len(issues) * 2 + len(sus) * 1.5 + len(misread) * 0.5
    return dict(ok=bool(ok), score=round(float(score), 3), wer=round(float(wer), 3), thr=round(thr, 3),
                heard=" ".join(w["w"] for w in raw), words=len(exp), issues=issues, passed_words=bool(passed),
                stress_checked=checked, suspicious=sus, misread=misread, stress_ok=bool(stress_ok),
                stress_tool="meter2" if have_meter else None)


def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

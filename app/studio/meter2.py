"""meter2: which syllable of each word carries the stress, from audio and the known text.

Port of meter/meter2.py from huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf (Apache-2.0) for RVC Studio:
  * models are loaded from the studio's models folder (models/wav2vec2-ru, models/meter2/meter/head.pt);
  * the device is chosen automatically (CUDA / MPS / XPU / CPU);
  * torchaudio.functional.forced_align is replaced by a small CTC Viterbi aligner (torchaudio is not released for
    current PyTorch any more) - same output: the token of every frame and its log-probability.

Pipeline: the text is known, so no ASR. A Russian character CTC model (jonatasgrosman/wav2vec2-large-xlsr-53-russian)
is forced-aligned to the letters; hidden states of layers named by the head are mean-pooled over each vowel span;
a small head (head.pt, trained on natural RuLS speech + Silero renders with deliberately moved stress) scores every
vowel; softmax over a word's vowels -> vowel_scores, best = argmax, margin = p(best) - p(runner-up).
Exact-syllable accuracy (author's numbers): 98.3 % on Russian LibriSpeech, 99.3 % on Silero renders.
The detector head was trained partly on Silero renders (non-commercial licence): a research tool.
"""
import json
import math
import os
import re
import threading
import warnings

import numpy as np

from . import device as D
from . import models as M

V = "аеёиоуыэюя"
ACUTE = "́"
WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
OTHER_RE = re.compile(r"[A-Za-z0-9]+(?:[.,'][A-Za-z0-9]+)*")
CONF_MIN = 0.5
FRAME = 0.02
SR = 16000


def plain_text(text):
    return text.replace(ACUTE, "").replace("+", "")


def words_in(text):
    t = plain_text(text)
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(t)]


def vowel_pos(word):
    return [i for i, ch in enumerate(word.lower()) if ch in V]


def marked_index(word_with_mark):
    """Vowel index carrying U+0301 (after the vowel) or '+' (before it), else None."""
    n, hit, idx = 0, False, None
    w = word_with_mark.lower()
    for j, ch in enumerate(w):
        if ch == "+":
            hit = True
            continue
        if ch in V:
            if hit or (j + 1 < len(w) and w[j + 1] == ACUTE):
                idx = n
            hit = False
            n += 1
    return idx


# --------------------------------------------------------------------------- models
class _Models:
    lock = threading.Lock()
    ctc = proc = vocab = None
    head = None
    device = "cpu"


def available():
    return M.installed("w2v_ru") and M.installed("stress_meter")


def load(want="auto", log=print):
    if _Models.ctc is None:
        with _Models.lock:
            if _Models.ctc is None:
                import torch
                from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
                M.ensure("w2v_ru", log=log)
                M.ensure("stress_meter", log=log)
                d = D.detect(want)
                dev = d["device"] if d["kind"] in ("cuda", "xpu", "mps") else "cpu"
                path = M.repo_dir("w2v_ru")
                try:
                    m = Wav2Vec2ForCTC.from_pretrained(path).eval().to(dev)
                except Exception:
                    dev = "cpu"
                    m = Wav2Vec2ForCTC.from_pretrained(path).eval()
                _Models.proc = Wav2Vec2FeatureExtractor.from_pretrained(path)
                _Models.vocab = json.load(open(os.path.join(path, "vocab.json"), encoding="utf-8"))
                ck = torch.load(os.path.join(M.repo_dir("stress_meter"), "meter", "head.pt"), map_location="cpu",
                                weights_only=False)
                h = _head_class()(ck["D"], ck["model"], ck["hidden"], ck["drop"], ck["nctx"])
                h.load_state_dict(ck["state"])
                h.eval()
                ck["net"] = h
                ck["feats"] = [tuple(f) for f in ck["feats"]]
                _Models.head = ck
                _Models.device = dev
                _Models.ctc = m
                log(f"  проверка ударений: meter2 на {'GPU' if dev != 'cpu' else 'CPU'}")
    return _Models.ctc


def free():
    with _Models.lock:
        _Models.ctc = None
        _Models.head = None


def to16k(wav, sr):
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1) if x.shape[1] < x.shape[0] else x.mean(axis=0)
    x = np.nan_to_num(x)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x.astype(np.float32)


# --------------------------------------------------------------------------- alignment
def ctc_forward(y, layers=()):
    import torch
    m = _Models.ctc
    inp = _Models.proc(y, sampling_rate=SR, return_tensors="pt")
    with torch.inference_mode():
        o = m(inp.input_values.to(_Models.device), output_hidden_states=bool(layers))
    lp = torch.log_softmax(o.logits[0].float(), -1).cpu()
    hs = {l: o.hidden_states[l][0].float().cpu().numpy() for l in layers} if layers else {}
    return lp, hs


def forced_align(em, ids, blank=0):
    """CTC Viterbi alignment (replacement for torchaudio.functional.forced_align).
    em: (T, C) log-probs, ids: target ids. -> (path (T,) token per frame incl. blanks, scores (T,) log-prob)."""
    em = np.asarray(em, np.float64)
    T = em.shape[0]
    z = [blank]
    for t in ids:
        z += [t, blank]
    z = np.array(z)
    S = len(z)
    NEG = -1e30
    alpha = np.full(S, NEG)
    alpha[0] = em[0, z[0]]
    if S > 1:
        alpha[1] = em[0, z[1]]
    back = np.zeros((T, S), np.int8)          # 0: stay, 1: from s-1, 2: from s-2
    can_skip = np.zeros(S, bool)
    can_skip[2:] = (z[2:] != blank) & (z[2:] != z[:-2])
    emz = em[:, z]
    for t in range(1, T):
        a0 = alpha
        a1 = np.r_[NEG, alpha[:-1]]
        a2 = np.where(can_skip, np.r_[NEG, NEG, alpha[:-2]], NEG)
        stack = np.stack([a0, a1, a2])
        k = np.argmax(stack, 0)
        alpha = stack[k, np.arange(S)] + emz[t]
        back[t] = k
    if S > 1 and alpha[S - 2] > alpha[S - 1]:
        s = S - 2
    else:
        s = S - 1
    if alpha[s] <= NEG / 2:
        raise RuntimeError("alignment impossible")
    path = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= back[t, s]
    tokens = z[path]
    scores = em[np.arange(T), tokens]
    return tokens, scores


def _targets(text):
    vocab = _Models.vocab
    t = plain_text(text)
    toks = []
    for m in re.finditer(WORD_RE.pattern + "|" + OTHER_RE.pattern, t):
        toks.append(("w" if WORD_RE.fullmatch(m.group(0)) else "x", m.group(0)))
    ids, units = [], []
    star = len(vocab)
    for kind, w in toks:
        if ids:
            ids.append(vocab["|"])
        if kind == "x":
            units.append((kind, w, [len(ids)]))
            ids.append(star)
            continue
        pos = []
        for ch in w.lower():
            if ch == "-":
                pos.append(None)
                continue
            ids.append(vocab.get(ch, star))
            pos.append(len(ids) - 1)
        units.append((kind, w, pos))
    return ids, units


def align(lp, text):
    import torch
    ids, units = _targets(text)
    nw = sum(1 for k, _, _ in units if k == "w")
    if not ids:
        return [None] * nw
    T = lp.shape[0]
    nonblank = torch.logsumexp(lp[:, 1:], dim=-1, keepdim=True) - 1.0
    em = torch.cat([lp, nonblank], dim=1).numpy()
    rep = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    if T < len(ids) + rep:
        return [None] * nw
    try:
        ali, sc = forced_align(em, ids, blank=0)
    except Exception:
        return [None] * nw
    ali, sc = ali.tolist(), sc.tolist()
    first, last = [None] * len(ids), [None] * len(ids)
    lsum, lcnt = [0.0] * len(ids), [0] * len(ids)
    j, prev = -1, 0
    for t, tok in enumerate(ali):
        if tok == 0:
            prev = 0
            continue
        if tok != prev or j < 0:
            j += 1
            while j < len(ids) and ids[j] != tok:
                j += 1
            if j >= len(ids):
                return [None] * nw
            first[j] = t
        last[j] = t
        lsum[j] += sc[t]
        lcnt[j] += 1
        prev = tok
    if any(f is None for f in first):
        return [None] * nw
    starts = first + [T]
    out = []
    for kind, w, pos in units:
        if kind != "w":
            continue
        real = [p for p in pos if p is not None]
        f0, f1 = first[real[0]], last[real[-1]] + 1
        letters = []
        for p in pos:
            if p is None:
                letters.append(None)
                continue
            nxt = starts[p + 1]
            if p == real[-1]:
                nxt = min(nxt, last[p] + 1 + 6)
            letters.append((first[p], max(nxt, first[p] + 1)))
        f1 = max(f1, letters[-1][1] if letters[-1] else f1)
        lconf = [math.exp(lsum[p] / max(lcnt[p], 1)) for p in real]
        out.append({"span": (f0, f1), "letters": letters, "conf": sum(lconf) / len(lconf), "lconf": lconf})
    return out


# --------------------------------------------------------------------------- head
def _head_class():
    import torch

    class Head(torch.nn.Module):
        """per-vowel MLP over [x, x - word mean] (as meter2's train_head.py)."""

        def __init__(self, d, kind="mlp", hidden=256, drop=0.2, nctx=0):
            super().__init__()
            self.kind = kind
            self.norm = torch.nn.LayerNorm(d)
            if kind == "linear":
                self.f = torch.nn.Linear(d, 1)
            else:
                self.proj = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(2 * d, hidden),
                                                torch.nn.GELU(), torch.nn.Dropout(drop))
                self.ctx = None
                if nctx:
                    layer = torch.nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout=drop, batch_first=True)
                    self.ctx = torch.nn.TransformerEncoder(layer, nctx)
                self.out = torch.nn.Linear(hidden, 1)

        def forward(self, X, Mk):
            X = self.norm(X)
            if self.kind == "linear":
                s = self.f(X).squeeze(-1)
            else:
                mf = Mk.unsqueeze(-1).float()
                mean = (X * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
                h = self.proj(torch.cat([X, X - mean], -1))
                if self.ctx is not None:
                    h = self.ctx(h, src_key_padding_mask=~Mk)
                s = self.out(h).squeeze(-1)
            return s.masked_fill(~Mk, -1e9)
    return Head


def _empty(words):
    return [{"word": w, "word_index": i, "vowel_scores": [] if not vowel_pos(w) else None,
             "best": None, "margin": None, "start": None, "end": None, "align": None}
            for i, (w, _, _) in enumerate(words)]


def judge(wav, sr, text, want="auto", log=print):
    """Per Cyrillic word of `text`: which vowel the audio stresses (see module doc). Never raises."""
    words = words_in(text)
    if not words:
        return []
    try:
        load(want, log)
        y = to16k(wav, sr)
        if y.size < int(0.2 * SR) or not np.isfinite(y).all() or np.abs(y).max() < 1e-5:
            return _empty(words)
        hd = _Models.head
        lc = sorted({l for k, l in hd["feats"] if k == "ctc"})
        lp, hc = ctc_forward(y, lc)
        al = align(lp, text)
        H = {("ctc", l): hc[l] for l in lc}
        return _score(hd, H, lp.shape[0], words, al)
    except Exception as e:
        warnings.warn(f"meter2.judge failed: {type(e).__name__}: {e}")
        return _empty(words)


def _score(head, H, T, words, al):
    import torch
    feats = head["feats"]
    res = _empty(words)
    todo, rows = [], []
    for i, ((w, _, _), a) in enumerate(zip(words, al)):
        vp = vowel_pos(w)
        if a is None:
            continue
        r = res[i]
        r["start"], r["end"] = round(a["span"][0] * FRAME, 3), round(a["span"][1] * FRAME, 3)
        r["align"] = round(a["conf"], 3)
        if not vp or a["conf"] < CONF_MIN:
            continue
        if len(vp) == 1:
            r.update(vowel_scores=[1.0], best=0, margin=1.0)
            continue
        V_ = []
        for ci in vp:
            s, e = a["letters"][ci]
            s = min(s, T - 1)
            e = min(max(e, s + 1), T)
            V_.append(np.concatenate([H[f][s:e].mean(0).astype(np.float16).astype(np.float32) for f in feats]))
        todo.append(i)
        rows.append(np.stack(V_))
    if rows:
        mv = max(len(v) for v in rows)
        X = np.zeros((len(rows), mv, rows[0].shape[1]), np.float32)
        Mk = np.zeros((len(rows), mv), bool)
        for j, v in enumerate(rows):
            X[j, :len(v)] = v
            Mk[j, :len(v)] = True
        with torch.inference_mode():
            sc = head["net"]((torch.tensor(X) - head["mu"]) / head["sd"], torch.tensor(Mk))
            P = torch.softmax(sc.float(), -1).numpy()
        for j, i in enumerate(todo):
            p = P[j, :Mk[j].sum()].astype(float)
            o = np.argsort(-p)
            res[i].update(vowel_scores=[round(float(v), 4) for v in p], best=int(o[0]),
                          margin=round(float(p[o[0]] - p[o[1]]), 4))
    return res

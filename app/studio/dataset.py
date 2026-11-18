"""Clean voice -> RVC training slices.

    remove DC + rumble -> (optional) spectral noise gate -> find where the voice is -> breaths keep/soften/remove
    -> cut out pauses -> slices of min..max seconds at natural pauses -> quality score per slice
    -> level normalisation -> mono WAV at the target sample rate.

Everything here is plain numpy/scipy (+ WORLD for voicing / pitch-shifted copies), works the same on any machine.
"""
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

FR = 0.01          # analysis hop, s
WIN = 0.03         # analysis window, s


# ---------------------------------------------------------------- helpers
def to_mono(x):
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        return x
    if x.shape[0] in (1, 2) and x.shape[0] < x.shape[1]:
        x = x.T
    return x.mean(1).astype(np.float32)


def resample(y, sr_in, sr_out):
    if sr_in == sr_out:
        return y.astype(np.float32)
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(y, sr_out // g, sr_in // g).astype(np.float32)


def frame_db(m, sr, hop=FR, win=WIN):
    h, w = int(sr * hop), int(sr * win)
    n = max(1, 1 + (len(m) - 1) // h)
    pad = np.pad(m, (w // 2, w + h))
    idx = np.arange(n)[:, None] * h + np.arange(0, w, 4)[None, :]       # decimated window: fast and good enough
    return 10 * np.log10(np.mean(pad[idx] ** 2, 1) + 1e-12)


def voicing(m, sr, hop=FR):
    """f0 per frame (0 = unvoiced) with WORLD dio + stonemask at 16 kHz."""
    import pyworld as pw
    y = resample(m, sr, 16000).astype(np.float64)
    f0, t = pw.dio(y, 16000, f0_floor=65, f0_ceil=1100, frame_period=hop * 1000)
    f0 = pw.stonemask(y, f0, t, 16000)
    return f0


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def highpass(m, sr, hz=70):
    sos = butter(4, hz, "highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, m).astype(np.float32)


def spectral_gate(m, sr, noise_mask=None, strength="normal"):
    """Stationary noise reduction: per-bin noise profile from the quiet frames, soft Wiener-like gain."""
    import librosa
    alpha, floor = {"light": (1.0, 0.2), "normal": (1.5, 0.1), "strong": (2.2, 0.05)}[strength]
    n_fft, hop = 2048, 512
    S = librosa.stft(m, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    if noise_mask is not None:
        fm = np.interp(np.arange(mag.shape[1]) * hop / sr, np.arange(len(noise_mask)) * FR, noise_mask.astype(float)) > 0.5
    else:
        fm = np.zeros(mag.shape[1], bool)
    if fm.sum() < 20:                                   # not enough pauses: quietest 10 % of frames
        e = mag.sum(0)
        fm = e <= np.percentile(e, 10)
    noise = np.percentile(mag[:, fm], 60, axis=1)[:, None]
    gain = np.clip(1 - alpha * noise / (mag + 1e-9), floor, 1.0)
    from scipy.ndimage import uniform_filter
    gain = uniform_filter(gain, size=(3, 5))            # smooth over freq/time: no "musical noise"
    y = librosa.istft(S * gain, hop_length=hop, length=len(m))
    return y.astype(np.float32)


# ---------------------------------------------------------------- analysis
def analyse(m, sr, log=print):
    db = frame_db(m, sr)
    f0 = voicing(m, sr)
    n = min(len(db), len(f0))
    db, f0 = db[:n], f0[:n]
    peak = np.percentile(db, 99.5)
    floor = np.percentile(db, 10)
    thr = max(floor + 10, peak - 38, -58)
    active = db > thr
    # hysteresis: extend runs while above thr-6
    lo = db > thr - 6
    for a, b in runs(active):
        i = a
        while i > 0 and lo[i - 1] and not active[i - 1]:
            i -= 1
            active[i] = True
        j = b
        while j < n and lo[j] and not active[j]:
            active[j] = True
            j += 1
    # fill very short gaps, drop tiny islands
    for a, b in runs(~active):
        if b - a < 8 and a > 0 and b < n:
            active[a:b] = True
    for a, b in runs(active):
        if b - a < 10:
            active[a:b] = False
    return dict(db=db, f0=f0, active=active, thr=float(thr), floor=float(floor), peak=float(peak))


def find_breaths(m, sr, an):
    """Breaths: unvoiced, noise-like, quieter than the singing, 0.15..1.2 s, not glued to a sibilant."""
    import librosa
    db, f0, act = an["db"], an["f0"], an["active"]
    voiced = (f0 > 0) & act
    if voiced.sum() < 20:
        return np.zeros_like(act)
    vlev = np.median(db[voiced])
    cent = librosa.feature.spectral_centroid(y=m, sr=sr, n_fft=1024, hop_length=int(sr * FR))[0]
    cent = np.pad(cent, (0, max(0, len(db) - len(cent))))[:len(db)]
    out = np.zeros_like(act)
    for a, b in runs(act & ~voiced):
        dur = (b - a) * FR
        if not (0.15 <= dur <= 1.2):
            continue
        lev = np.median(db[a:b])
        c = np.median(cent[a:b])
        if lev < vlev - 9 and c < 5500:
            out[a:b] = True
    return out


# ---------------------------------------------------------------- slicing
def plan_slices(an, breaths, breath_mode="soften", min_len=2.0, max_len=10.0, join_gap=0.35, keep_gap=0.2):
    """-> list of slices, each a list of (start_frame, end_frame) pieces to concatenate."""
    act = an["active"].copy()
    if breath_mode == "remove":
        act &= ~breaths
    pieces = [(a, b) for a, b in runs(act) if (b - a) * FR >= 0.12]
    if not pieces:
        return []
    # phrases = pieces separated by short gaps
    phrases, cur = [], [pieces[0]]
    for p in pieces[1:]:
        if (p[0] - cur[-1][1]) * FR < join_gap:
            cur.append(p)
        else:
            phrases.append(cur)
            cur = [p]
    phrases.append(cur)
    db = an["db"]
    maxf, minf = int(max_len / FR), int(min_len / FR)

    def length(ps):
        return sum(b - a for a, b in ps) + int(keep_gap / FR) * (len(ps) - 1)

    def split_long(ps):
        """a phrase longer than max_len: cut at the quietest frame in the middle part."""
        out, queue = [], [ps]
        while queue:
            q = queue.pop()
            if length(q) <= maxf:
                out.append(q)
                continue
            if len(q) > 1:   # cut between pieces first, at the widest gap near the middle
                total = length(q)
                acc, best, bk = 0, None, 1
                for k in range(1, len(q)):
                    acc += q[k - 1][1] - q[k - 1][0]
                    score = abs(acc - total / 2) - 3 * (q[k][0] - q[k - 1][1])
                    if best is None or score < best:
                        best, bk = score, k
                queue += [q[:bk], q[bk:]]
                continue
            a, b = q[0]
            lo, hi = a + (b - a) // 4, b - (b - a) // 4
            c = lo + int(np.argmin(db[lo:hi])) if hi > lo else (a + b) // 2
            queue += [[(a, c)], [(c, b)]]
        return sorted(out, key=lambda z: z[0][0])

    slices, cur = [], []
    for ph in phrases:
        for part in split_long(ph):
            if cur and length(cur + part) > maxf:
                slices.append(cur)
                cur = []
            cur = cur + part
            if length(cur) >= minf * 2:       # long enough: close it at this natural pause
                slices.append(cur)
                cur = []
    if cur:
        if slices and length(cur) < minf and length(slices[-1] + cur) <= maxf * 1.2:
            slices[-1] = slices[-1] + cur
        else:
            slices.append(cur)
    return slices


def render(m, sr, pieces, breaths=None, breath_mode="soften", keep_gap=0.2, pad=0.04):
    """Concatenate pieces with short equal gaps and fades; soften breaths if asked."""
    h = int(sr * FR)
    gap = np.zeros(int(sr * keep_gap), np.float32)
    out = []
    for k, (a, b) in enumerate(pieces):
        s0, s1 = max(0, a * h - int(pad * sr)), min(len(m), b * h + int(pad * sr))
        seg = m[s0:s1].copy()
        if breaths is not None and breath_mode == "soften":
            g = np.ones(len(seg), np.float32)
            for ba, bb in runs(breaths[a:b]):
                i0, i1 = (ba + a) * h - s0, (bb + a) * h - s0
                g[max(0, i0):max(0, i1)] = 0.25          # -12 dB
            from scipy.ndimage import uniform_filter1d
            seg *= uniform_filter1d(g, int(0.03 * sr))
        f = min(int(0.015 * sr), len(seg) // 4)
        if f > 0:
            seg[:f] *= np.linspace(0, 1, f)
            seg[-f:] *= np.linspace(1, 0, f)
        if k:
            out.append(gap)
        out.append(seg)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ---------------------------------------------------------------- quality
def quality(y, sr):
    import librosa
    n = len(y)
    dur = n / sr
    if n < sr * 0.3:
        return dict(score=0, dur=round(dur, 2), flags=["слишком короткий"])
    db = frame_db(y, sr)
    f0 = voicing(y, sr)[:len(db)]
    db = db[:len(f0)]
    loud = db > db.max() - 30
    vr = float(((f0 > 0) & loud).sum() / max(1, loud.sum()))
    flat = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=512)[0]
    fl = float(np.median(flat))
    clip = float(np.mean(np.abs(y) > 0.985))
    lev = float(10 * np.log10(np.mean(y ** 2) + 1e-12))
    flags, score = [], 100.0
    if dur < 1.5:
        flags.append("короткий")
        score -= 25
    if vr < 0.35:
        flags.append("мало голоса")
        score -= 40 * (0.35 - vr) / 0.35 + 15
    if fl > 0.08:
        flags.append("шумный")
        score -= min(35, (fl - 0.08) * 400)
    if clip > 0.001:
        flags.append("перегруз")
        score -= min(30, clip * 3000)
    if lev < -42:
        flags.append("тихий")
        score -= 10
    return dict(score=int(max(0, min(100, score))), dur=round(dur, 2), voiced=round(vr, 2), flatness=round(fl, 3),
                clip=round(clip, 4), level_db=round(lev, 1), flags=flags)


def normalise(y, target_db=-20.0, peak_db=-1.0):
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    g = 10 ** (target_db / 20) / rms
    pk = np.abs(y).max() * g + 1e-12
    lim = 10 ** (peak_db / 20)
    if pk > lim:
        g *= lim / pk
    return (y * g).astype(np.float32)


def write(path, y, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.clip(y, -1, 1), sr, subtype="PCM_16")


# ---------------------------------------------------------------- pitch-shifted copies (augmentation)
def world_analyse(y, sr):
    import pyworld as pw
    x = y.astype(np.float64)
    f0, t = pw.dio(x, sr, f0_floor=65, f0_ceil=1100, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    return f0, pw.cheaptrick(x, f0, t, sr), pw.d4c(x, f0, t, sr)


def pitch_shift_world(y, sr, semis, ana=None):
    """Pitch shift that keeps the formants (timbre): WORLD analysis, f0 * 2^(s/12), same spectral envelope."""
    import pyworld as pw
    f0, sp, ap = ana or world_analyse(y, sr)
    z = pw.synthesize(f0 * 2 ** (semis / 12), sp, ap, sr, 5.0)
    z = np.pad(z, (0, max(0, len(y) - len(z))))[:len(y)]
    return z.astype(np.float32)


# ---------------------------------------------------------------- main entry
def build(x, sr, out_dir, prefix, opts, log=print, progress=None, skip_detect=False):
    """x: voice (n,) or (n, ch) at sr. Writes slices into out_dir; returns dict(slices=[...], stats=...).
    opts: denoise off|light|normal|strong, breaths keep|soften|remove, min_len, max_len, target_sr, level_db."""
    m = to_mono(x)
    m = highpass(m, sr, 70)
    m -= np.mean(m)
    if progress:
        progress(0.05, "анализ: где голос, где паузы")
    an = analyse(m, sr, log)
    if opts.get("denoise", "light") != "off":
        if progress:
            progress(0.15, "шумоподавление")
        m = spectral_gate(m, sr, ~an["active"], opts.get("denoise", "light"))
        an = analyse(m, sr, log)
    breaths = find_breaths(m, sr, an)
    bmode = opts.get("breaths", "soften")
    sl = plan_slices(an, breaths, bmode, float(opts.get("min_len", 2.0)), float(opts.get("max_len", 10.0)))
    tsr = int(opts.get("target_sr", 40000))
    level = float(opts.get("level_db", -20))
    min_score = int(opts.get("min_score", 45))
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".wav") and f.startswith(prefix + "_"):
            os.remove(os.path.join(out_dir, f))
    res = []
    for k, pieces in enumerate(sl):
        y = render(m, sr, pieces, breaths, bmode)
        q = quality(y, sr)
        y = normalise(resample(y, sr, tsr), level)
        name = f"{prefix}_{k + 1:04d}.wav"
        write(os.path.join(out_dir, name), y, tsr)
        res.append(dict(file=name, src_start=round(pieces[0][0] * FR, 2), src_end=round(pieces[-1][1] * FR, 2),
                        include=q["score"] >= min_score and q["dur"] >= 1.0, **q))
        if progress:
            progress(0.2 + 0.8 * (k + 1) / len(sl), f"нарезка {k + 1}/{len(sl)}")
    act_s = float(an["active"].sum() * FR)
    stats = dict(source_s=round(len(m) / sr, 1), voice_s=round(act_s, 1), breaths_s=round(float(breaths.sum() * FR), 1),
                 slices=len(res), included=sum(r["include"] for r in res),
                 dataset_s=round(sum(r["dur"] for r in res if r["include"]), 1), target_sr=tsr)
    return dict(slices=res, stats=stats)


def augment(src_dir, slices, out_dir, prefix, semis_list, target_sr, log=print, progress=None):
    """Formant-preserving pitch-shifted copies of the included slices."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".wav"):
            os.remove(os.path.join(out_dir, f))
    out = []
    todo = [s for s in slices if s.get("include")]
    for k, s in enumerate(todo):
        y, sr = sf.read(os.path.join(src_dir, s["file"]), dtype="float32")
        ana = world_analyse(y, sr)
        for st in semis_list:
            z = normalise(pitch_shift_world(y, sr, st, ana), 10 * np.log10(np.mean(y ** 2) + 1e-12))
            name = f"{prefix}_{os.path.splitext(s['file'])[0].split('_')[-1]}_{'+' if st > 0 else ''}{st:g}st.wav"
            write(os.path.join(out_dir, name), z, sr)
            out.append(dict(file=name, src=s["file"], semis=st, dur=s["dur"], include=True))
        if progress:
            progress((k + 1) / max(1, len(todo)), f"копии со сдвигом высоты {k + 1}/{len(todo)}")
    return out


def concat_all(src_dir, files, out_path, sr, gap=0.4):
    """All included slices in one file (some RVC forks prefer a single long WAV)."""
    parts = []
    z = np.zeros(int(sr * gap), np.float32)
    for f in files:
        y, s = sf.read(os.path.join(src_dir, f), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        parts += [resample(y, s, sr), z]
    if parts:
        write(out_path, np.concatenate(parts), sr)


if __name__ == "__main__":
    import sys
    x, sr = sf.read(sys.argv[1], dtype="float32")
    r = build(x, sr, sys.argv[2], "clean", dict(denoise="light", breaths="soften"))
    print(json.dumps(r["stats"], ensure_ascii=False, indent=1))

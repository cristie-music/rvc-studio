"""Audio file helpers: read anything libsndfile knows (+ audioread/ffmpeg fallback), resample, write WAV/MP3, peaks."""
import io
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 44100
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".oga", ".aif", ".aiff", ".m4a", ".aac", ".opus", ".wma")


def read_any(path):
    """-> (float32 array (n, ch), sr). WAV/FLAC/MP3/OGG/AIFF via libsndfile; WAV that wraps an Ogg stream (FL samples);
    other formats (m4a/aac/...) through librosa/audioread if ffmpeg is installed."""
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except Exception as e1:
        data = open(path, "rb").read()
        i = data.find(b"OggS")
        if 0 <= i < 4096:
            return sf.read(io.BytesIO(data[i:]), dtype="float32", always_2d=True)
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=False)
            y = np.atleast_2d(y).T.astype(np.float32)
            return y, sr
        except Exception:
            raise RuntimeError(f"не удалось прочитать {os.path.basename(path)}: {e1}. "
                               "Поддерживаются WAV, FLAC, MP3, OGG, AIFF (для M4A/AAC нужен ffmpeg).")


def load_stereo(path, sr=SR):
    x, sr_in = read_any(path)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    x = x[:, :2]
    if sr_in != sr:
        g = np.gcd(sr, sr_in)
        x = resample_poly(x, sr // g, sr_in // g, axis=0).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32), sr_in


def write_wav(path, x, sr=SR, subtype="PCM_24"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > 1.0 and subtype != "FLOAT":   # separated stems may exceed 0 dBFS a little: keep them unclipped
        subtype = "FLOAT"
    sf.write(path, x, sr, subtype=subtype)


def write_mp3(path, x, sr=SR):
    """Small preview for the browser. Written in blocks (libsndfile's encoders crash on one huge write)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.clip(x, -1, 1).astype(np.float32)
    if y.ndim == 1:
        y = y[:, None]
    with sf.SoundFile(path + ".tmp", "w", sr, y.shape[1], format="MP3", subtype="MPEG_LAYER_III") as f:
        for i in range(0, len(y), 16384):
            f.write(y[i:i + 16384])
    os.replace(path + ".tmp", path)


def shift(x, s, sr=SR):
    """Move audio by s seconds: s > 0 adds silence in front, s < 0 cuts the start."""
    k = int(round(s * sr))
    if k > 0:
        return np.concatenate([np.zeros((k,) + x.shape[1:], x.dtype), x])
    if k < 0:
        return x[-k:]
    return x


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def peaks(x, points=1600):
    """Waveform overview: max |amplitude| per column (0..1), mono."""
    m = np.abs(x).max(1) if x.ndim == 2 else np.abs(x)
    n = max(1, len(m) // points)
    m = m[: n * (len(m) // n)].reshape(-1, n).max(1)
    return [round(float(v), 3) for v in m]


def peaks_cached(path, points=1600):
    cache = path + ".peaks.json"
    try:
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
            return json.load(open(cache))
    except OSError:
        pass
    x, sr = read_any(path)
    d = dict(peaks=peaks(x, points), duration=len(x) / sr)
    try:
        json.dump(d, open(cache, "w"))
    except OSError:
        pass
    return d


def activity(x, sr=SR, block=1.0, rel_db=-32.0, ref_db=None):
    """Seconds where the stem really plays: blocks louder than (reference level + rel_db) and above -60 dBFS."""
    m = x.mean(1) if x.ndim == 2 else x
    n = int(block * sr)
    k = len(m) // n
    if k == 0:
        return np.zeros(0, bool), []
    lv = 20 * np.log10(np.sqrt(np.mean(m[:k * n].reshape(k, n) ** 2, 1)) + 1e-12)
    thr = max(-60.0, (ref_db if ref_db is not None else lv.max()) + rel_db)
    return lv > thr, [round(float(v), 1) for v in lv]

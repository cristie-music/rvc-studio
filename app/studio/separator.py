"""Mel-Band RoFormer inference (vocals, karaoke, de-echo, de-reverb) in pure PyTorch.

Network code: lucidrains "BS-RoFormer" (Mel-Band variant) + ZFTurbo "Music-Source-Separation-Training" (MIT license),
adapted from Guitar Splitter's roformer.py to return every stem a checkpoint was trained for and to work on any device
(CUDA / ROCm / MPS / XPU / CPU) with automatic fallback when video memory runs out.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from . import device as D
from . import models as M

SR = 44100


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Rotary(nn.Module):
    """rotary-embedding-torch RotaryEmbedding(dim_head), 'lang' frequencies, interleaved pairs."""

    def __init__(self, dim):
        super().__init__()
        self.freqs = nn.Parameter(1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)), requires_grad=False)
        self._cache = {}

    def rotate(self, t):
        n, dev = t.shape[-2], t.device
        key = (n, dev, t.dtype)
        if key not in self._cache:
            f = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * self.freqs.float()[None]
            f = f.repeat_interleave(2, -1)
            self._cache = {key: (f.cos().to(t.dtype), f.sin().to(t.dtype))}
        cos, sin = self._cache[key]
        x = t.reshape(*t.shape[:-1], -1, 2)
        x1, x2 = x.unbind(-1)
        rot = torch.stack((-x2, x1), -1).reshape(t.shape)
        return t * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, rotary):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.rotary_embed = rotary
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False))

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.rotary_embed.rotate(q), self.rotary_embed.rotate(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out * self.to_gates(x).transpose(1, 2).unsqueeze(-1).sigmoid()
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(), nn.Identity(),
                                 nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, rotary, norm_output=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, dim_head, rotary), FeedForward(dim)])
                                     for _ in range(depth)])
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs])

    def forward(self, x):
        return torch.stack([f(s) for s, f in zip(x.split(self.dim_inputs, -1), self.to_features)], -2)


def _mlp(dim_in, dim_out, dim_hidden, depth):
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs, depth, mult):
        super().__init__()
        self.to_freqs = nn.ModuleList([nn.Sequential(_mlp(dim, d * 2, dim * mult, depth), nn.GLU(-1))
                                       for d in dim_inputs])

    def forward(self, x):
        return torch.cat([f(b) for b, f in zip(x.unbind(-2), self.to_freqs)], -1)


class MelBandRoformer(nn.Module):
    def __init__(self, m, norm_output=False):
        super().__init__()
        import librosa
        dim, self.ch = m["dim"], 2 if m["stereo"] else 1
        t_rot, f_rot = Rotary(m["dim_head"]), Rotary(m["dim_head"])
        self.layers = nn.ModuleList([nn.ModuleList([
            Transformer(dim, m["time_transformer_depth"], m["heads"], m["dim_head"], t_rot, norm_output),
            Transformer(dim, m["freq_transformer_depth"], m["heads"], m["dim_head"], f_rot, norm_output)])
            for _ in range(m["depth"])])
        self.stft_kw = dict(n_fft=m["stft_n_fft"], hop_length=m["stft_hop_length"], win_length=m["stft_win_length"],
                            normalized=m["stft_normalized"])
        n_freqs = m["stft_n_fft"] // 2 + 1
        fb = torch.from_numpy(librosa.filters.mel(sr=m["sample_rate"], n_fft=m["stft_n_fft"], n_mels=m["num_bands"]))
        fb[0][0] = 1.0
        fb[-1, -1] = 1.0
        per_band = fb > 0
        idx = torch.arange(n_freqs).repeat(m["num_bands"], 1)[per_band]
        if self.ch == 2:
            idx = (idx[:, None] * 2 + torch.arange(2)).reshape(-1)
        self.register_buffer("freq_indices", idx, persistent=False)
        self.register_buffer("num_bands_per_freq", per_band.sum(0), persistent=False)
        dims = tuple(2 * int(f) * self.ch for f in per_band.sum(1).tolist())
        self.band_split = BandSplit(dim, dims)
        self.mask_estimators = nn.ModuleList([MaskEstimator(dim, dims, m["mask_estimator_depth"],
                                                            m.get("mlp_expansion_factor", 4))
                                              for _ in range(m["num_stems"])])

    def forward(self, audio):
        """audio (b, ch, n) -> every stem (b, stems, ch, n)."""
        b, ch, n = audio.shape
        win = torch.hann_window(self.stft_kw["win_length"], device=audio.device)
        spec = torch.stft(audio.float().reshape(b * ch, n), **self.stft_kw, window=win, return_complex=True)
        spec = torch.view_as_real(spec)                                     # (b*ch, f, t, 2)
        f, t = spec.shape[1:3]
        spec = spec.reshape(b, ch, f, t, 2).permute(0, 2, 1, 3, 4).reshape(b, f * ch, t, 2)  # (f s) interleaved
        x = spec[:, self.freq_indices].permute(0, 2, 1, 3).reshape(b, t, -1)
        x = self.band_split(x)                                               # (b, t, bands, d)
        for time_tf, freq_tf in self.layers:
            nb, d = x.shape[2], x.shape[3]
            x = x.permute(0, 2, 1, 3).reshape(b * nb, t, d)
            x = time_tf(x).reshape(b, nb, t, d).permute(0, 2, 1, 3).reshape(b * t, nb, d)
            x = freq_tf(x).reshape(b, t, nb, d)
        cspec = torch.view_as_complex(spec.float().contiguous())             # (b, f*ch, t)
        denom = self.num_bands_per_freq.repeat_interleave(ch).clamp(min=1).float()
        outs = []
        for est in self.mask_estimators:
            mask = est(x)                                                    # (b, t, F*2)
            mask = torch.view_as_complex(mask.reshape(b, t, -1, 2).permute(0, 2, 1, 3).float().contiguous())
            summed = torch.zeros_like(cspec).scatter_add_(1, self.freq_indices[None, :, None].expand(b, -1, t), mask)
            o = cspec * (summed / denom[None, :, None])
            o = o.reshape(b, f, ch, t).permute(0, 2, 1, 3).reshape(b * ch, f, t)
            o[:, 0] = 0.0                                                    # zero DC, as in training
            outs.append(torch.istft(o, **self.stft_kw, window=win, length=n).reshape(b, ch, -1))
        return torch.stack(outs, 1)


def load(key, dev):
    spec = M.SEPARATION[key]
    cfg = yaml.load(open(M.path(spec["config"]), encoding="utf-8"), Loader=yaml.FullLoader)
    sd = torch.load(M.path(spec["ckpt"]), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.removeprefix("module."): v.float() for k, v in sd.items()}
    net = MelBandRoformer(cfg["model"], norm_output="layers.0.0.norm.gamma" in sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.endswith(("freq_indices", "num_bands_per_freq"))]
    if real_missing or unexpected:
        raise RuntimeError(f"модель {key}: не совпадает архитектура (нет {real_missing[:3]}, лишние {unexpected[:3]})")
    net.to(dev).eval()
    tr = cfg["training"]
    names = [s.lower() for s in tr["instruments"]]
    if cfg["model"]["num_stems"] == 1:
        names = [(tr.get("target_instrument") or names[0]).lower()]
    return net, cfg, names


def separate(audio, key, want_device="auto", overlap=2, progress=None, log=print):
    """audio float32 (2, n) at 44.1 kHz -> dict stem -> (2, n). For 1-stem models the complement is added as
    'rest' (= input - target). Windows of chunk_size with a linear cross-fade, several windows per batch on a GPU;
    out of memory -> smaller batch -> CPU."""
    dev, dinfo = D.torch_device(want_device)
    if dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    net, cfg, names = load(key, dev)
    batch = 1
    if dev.type == "cuda":
        vram = dinfo.get("vram_gb") or 4
        batch = 4 if vram >= 14 else 2 if vram >= 7 else 1
        torch.backends.cuda.matmul.allow_tf32 = True
    amp = dev.type == "cuda"
    chunk = cfg["audio"]["chunk_size"]
    hop = int(chunk / overlap)
    fade = chunk // 10
    n = audio.shape[1]
    border = chunk - hop
    mix = np.pad(audio.astype(np.float32), ((0, 0), (border, border + chunk)), mode="constant")
    win = np.ones(chunk, np.float32)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    out = np.zeros((len(names),) + mix.shape, np.float32)
    wsum = np.zeros(mix.shape[1], np.float32)
    starts = list(range(0, border + n + 1, hop))
    i, t0 = 0, time.time()
    with torch.inference_mode():
        while i < len(starts):
            ss = starts[i:i + batch]
            try:
                x = torch.from_numpy(np.stack([mix[:, s:s + chunk] for s in ss])).to(dev)
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        ys = net(x).float().cpu().numpy()
                else:
                    ys = net(x).float().cpu().numpy()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if dev.type == "cpu":
                    raise
                oom = "out of memory" in str(e).lower()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if batch > 1 and oom:
                    batch //= 2
                    log(f"  ! мало видеопамяти -> пакет {batch}")
                else:
                    log(f"  ! ускоритель не справился ({str(e)[:120]}) -> продолжаю на CPU")
                    dev, amp = torch.device("cpu"), False
                    torch.set_num_threads(max(1, os.cpu_count() or 1))
                    net.to(dev)
                continue
            for s, y in zip(ss, ys):
                w = win.copy()
                if s == 0:
                    w[:fade] = 1
                out[:, :, s:s + chunk] += y * w
                wsum[s:s + chunk] += w
            i += len(ss)
            if progress:
                el = time.time() - t0
                progress(i / len(starts), f"{i}/{len(starts)} окон, осталось ~{el / i * (len(starts) - i):.0f} с")
    if dev.type == "cuda":
        del net
        torch.cuda.empty_cache()
    out /= np.maximum(wsum, 1e-6)
    out = out[:, :, border:border + n]
    stems = {nm: out[k] for k, nm in enumerate(names)}
    if len(names) == 1:
        stems["rest"] = audio.astype(np.float32) - out[0]
    return stems, dict(device=D.label(dinfo), seconds=round(time.time() - t0, 1), model=M.SEPARATION[key]["title"])

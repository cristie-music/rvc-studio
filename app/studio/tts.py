"""Qwen3-TTS voice cloning (siriusfreak ru-stress-cf: Qwen3-TTS-12Hz-1.7B-Base fine-tuned to follow stress marks).

The reference = a clean piece of the singer's voice + its transcript (in-context cloning). One voice prompt is built
once and reused for every block. Device: CUDA (bf16/fp16) -> MPS -> XPU -> CPU (fp32), chosen automatically.
"""
import importlib.util
import os
import sys
import types

import numpy as np

from . import device as D
from . import models as M


def _stub_torchaudio():
    """qwen_tts imports torchaudio.compliance.kaldi for the old 25 Hz tokenizer only. torchaudio is not released for
    current PyTorch any more, so give it a stub when the real package is absent (the 12 Hz model never calls it)."""
    if "torchaudio" in sys.modules:
        return
    try:
        if importlib.util.find_spec("torchaudio") is not None:
            return
    except (ValueError, ImportError):
        pass
    # transformers checks for torchaudio once, when it is imported: let it see the truth (not installed) first
    import transformers.utils.import_utils  # noqa: F401
    from importlib.machinery import ModuleSpec
    ta = types.ModuleType("torchaudio")
    comp = types.ModuleType("torchaudio.compliance")
    kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    for m in (ta, comp, kaldi):       # a spec, so that importlib.util.find_spec() of other libraries does not raise
        m.__spec__ = ModuleSpec(m.__name__, None)

    def fbank(*a, **k):
        raise RuntimeError("torchaudio не установлен: нужен только для токенизатора 25 Гц")
    kaldi.fbank = fbank
    comp.kaldi = kaldi
    ta.compliance = comp
    sys.modules.update({"torchaudio": ta, "torchaudio.compliance": comp, "torchaudio.compliance.kaldi": kaldi})


def _stub_sox():
    """the 'sox' package (25 Hz tokenizer only) shells out to a SoX binary on import and prints errors: stub it."""
    if "sox" not in sys.modules:
        from importlib.machinery import ModuleSpec
        m = types.ModuleType("sox")
        m.__spec__ = ModuleSpec("sox", None)
        sys.modules["sox"] = m


class TTS:
    def __init__(self, key="tts", want_device="auto", log=print):
        import torch
        import logging
        logging.getLogger("sox").setLevel(logging.ERROR)     # the 'sox' package complains about a missing SoX binary
        _stub_torchaudio()
        _stub_sox()
        import warnings
        warnings.filterwarnings("ignore")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from qwen_tts import Qwen3TTSModel
        M.ensure(key, log=log)
        d = D.detect(want_device)
        self.dinfo = d
        kind = d["kind"]
        if kind == "cuda":
            # bf16 natively from Ampere (sm 8.x) on; older cards (T4, GTX 10/16, RTX 20) get float32 - fp16 can overflow
            major = torch.cuda.get_device_capability()[0] if not getattr(torch.version, "hip", None) else 9
            dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float32
            dmap = d["device"]
        elif kind == "xpu":
            dtype, dmap = torch.bfloat16, "xpu:0"
        elif kind == "mps":
            dtype, dmap = torch.float16, "mps"
        else:
            dtype, dmap = torch.float32, "cpu"
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        log(f"  озвучка: {M.ALL[key]['title']} на {D.label(d)} ({str(dtype).replace('torch.', '')})")
        try:
            self.model = Qwen3TTSModel.from_pretrained(M.repo_dir(key), device_map=dmap, dtype=dtype,
                                                       attn_implementation="sdpa")
        except Exception as e:
            if kind == "cpu":
                raise
            log(f"  ! ускоритель не справился ({str(e)[:160]}) -> CPU")
            self.dinfo = D.detect("cpu")
            self.model = Qwen3TTSModel.from_pretrained(M.repo_dir(key), device_map="cpu", dtype=torch.float32,
                                                       attn_implementation="sdpa")
        self.key = key
        self.prompt = None

    def set_voice(self, wav, sr, ref_text):
        """wav: mono float32. ref_text: what is said/sung in it ('' -> speaker embedding only, weaker)."""
        self.prompt = self.model.create_voice_clone_prompt(ref_audio=(wav.astype(np.float32), int(sr)),
                                                           ref_text=ref_text or None,
                                                           x_vector_only_mode=not ref_text)

    def embed(self, wav, sr):
        """speaker embedding (the model's own x-vector encoder), L2-normalised numpy vector."""
        import librosa
        inner = self.model.model
        tsr = inner.speaker_encoder_sample_rate
        w = wav.astype(np.float32)
        if sr != tsr:
            w = librosa.resample(w, orig_sr=int(sr), target_sr=tsr)
        e = inner.extract_speaker_embedding(audio=w, sr=tsr)
        e = e.float().cpu().numpy().reshape(-1) if hasattr(e, "cpu") else np.asarray(e, np.float32).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)

    def say(self, texts, seed=0, temperature=0.8, top_p=0.9):
        """texts: list of strings (stress marks as U+0301). -> (list of float32 arrays, sr)."""
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        longest = max(len(t) for t in texts)
        max_tokens = int(min(2048, (longest * 0.11 + 4) * 12 * 1.6))     # 12 codec frames per second
        wavs, sr = self.model.generate_voice_clone(text=list(texts), language=["Russian"] * len(texts),
                                                   voice_clone_prompt=self.prompt, temperature=temperature,
                                                   top_p=top_p, max_new_tokens=max_tokens)
        return [np.asarray(w, np.float32).reshape(-1) for w in wavs], int(sr)

    def close(self):
        import torch
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

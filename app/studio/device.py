"""Hardware auto-detection: which accelerator PyTorch can use, and how much memory there is.

Order for 'auto':  NVIDIA CUDA (also AMD ROCm builds, they report as cuda)  ->  Apple MPS  ->  Intel XPU  ->  CPU.
Everything works on CPU, accelerators only make the neural networks faster.
"""
import os
import platform
import subprocess


def _ram_gb():
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 2 ** 30, 1)
        if hasattr(os, "sysconf"):
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except Exception:
        pass
    return None


def _cpu_name():
    try:
        if os.name == "nt":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=3).stdout.strip() or platform.processor()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def detect(want="auto"):
    """-> dict(kind='cuda'|'mps'|'xpu'|'cpu', device='cuda:0'|..., name, vram_gb, ...)"""
    info = dict(kind="cpu", device="cpu", name=_cpu_name(), vram_gb=None, cpu=_cpu_name(),
                threads=os.cpu_count(), ram_gb=_ram_gb(), torch=None, backend="CPU", os=platform.platform(terse=True))
    try:
        import torch
    except Exception as e:  # no torch at all: only analysis/MIDI of already separated stems would work
        info["torch"] = f"нет PyTorch ({e})"
        return info
    info["torch"] = torch.__version__
    want = (want or "auto").lower()
    if want == "cpu":
        return info
    try:
        if want in ("auto", "cuda", "gpu") and torch.cuda.is_available():
            i = torch.cuda.current_device()
            p = torch.cuda.get_device_properties(i)
            rocm = getattr(torch.version, "hip", None)
            info.update(kind="cuda", device=f"cuda:{i}", name=p.name, vram_gb=round(p.total_memory / 2 ** 30, 1),
                        backend=f"ROCm {rocm}" if rocm else f"CUDA {torch.version.cuda}")
            return info
        if want in ("auto", "mps", "gpu") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            info.update(kind="mps", device="mps", name="Apple Silicon GPU", vram_gb=info["ram_gb"], backend="Metal (MPS)")
            return info
        if want in ("auto", "xpu", "gpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            p = torch.xpu.get_device_properties(0)
            info.update(kind="xpu", device="xpu:0", name=p.name, vram_gb=round(p.total_memory / 2 ** 30, 1),
                        backend="Intel XPU")
            return info
    except Exception as e:
        info["warning"] = f"ускоритель не инициализировался ({e}) -> CPU"
    return info


def torch_device(want="auto"):
    import torch
    d = detect(want)
    return torch.device(d["device"]), d


def label(d):
    if d["kind"] == "cpu":
        return f"CPU {d['name']} ({d['threads']} потоков)"
    v = f", {d['vram_gb']:.0f} GB" if d.get("vram_gb") else ""
    return f"GPU {d['name']} ({d['backend']}{v})"


if __name__ == "__main__":
    import json
    d = detect()
    print(json.dumps(d, ensure_ascii=False, indent=1))
    print(label(d))

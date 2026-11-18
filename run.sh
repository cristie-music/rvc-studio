#!/usr/bin/env bash
# RVC Studio (Linux/macOS): starts the web studio on http://127.0.0.1:8791/ ; Ctrl+C to stop.
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || bash ./install.sh
export HF_HOME="$(pwd)/models/hf_home" TORCH_HOME="$(pwd)/models/torch_home" PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
exec .venv/bin/python app/server.py "$@"

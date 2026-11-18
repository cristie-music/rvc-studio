"""Builds dist\\rvc-studio-app.zip: code only (app, engines, install.bat / install.sh, Colab notebook), ~1 MB.
Upload it to Colab, or unpack on another PC and run install.bat (Windows) / install.sh (Linux, macOS).

    python tools\\make_dist.py
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
INCLUDE_DIRS = ["app", "tools", "colab"]
INCLUDE_FILES = ["install.bat", "install.sh", "run.sh", "RVCStudio.bat", "dataset.bat", "harry.txt", "check-gpu.bat",
                 "requirements.txt", "README.txt", "README.md", "THIRD_PARTY_NOTICES.txt"]
SKIP_DIRS = {"__pycache__", "output", "_cache", ".ipynb_checkpoints"}


def build_zip(dst):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in INCLUDE_FILES:
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                z.write(p, f)
        for d in INCLUDE_DIRS:
            for root, dirs, files in os.walk(os.path.join(ROOT, d)):
                dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
                for f in files:
                    if f.endswith((".pyc", ".peaks.json")):
                        continue
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, ROOT))
    print("  ", dst, f"{os.path.getsize(dst) / 1e6:.1f} MB")


def main():
    os.makedirs(DIST, exist_ok=True)
    z = os.path.join(DIST, "rvc-studio-app.zip")
    build_zip(z)


if __name__ == "__main__":
    main()

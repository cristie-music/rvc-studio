# RVC Studio

**Turn a song (WAV / MP3) into a training dataset for an RVC voice model**, in your browser, locally.
The studio separates the vocal from the music, removes echo / delay / reverb, and builds a clean, sliced,
level-matched dataset for [Applio](https://github.com/IAHispano/Applio) or RVC WebUI. It also tells you **how many
epochs to train**.

Portable (everything lives in one folder), Python, auto-detects the hardware: NVIDIA CUDA, AMD ROCm (Linux),
Apple MPS, Intel XPU, or plain CPU. A Google Colab notebook (with ngrok support) is included.


## Pipeline

| Step | What happens | Model |
|---|---|---|
| 1. Vocal | vocal separated from the mix | Mel-Band RoFormer Vocals (becruily) |
| 2. Lead only | backing vocals, choirs, doubles removed (optional) | Mel-Band RoFormer Karaoke (becruily) |
| 3. Dry voice | delay/echo + reverb removed: off / reverb / echo+reverb / maximum | Dereverb-Echo MBR v2 (Sucial), De-Reverb MBR (anvuew) |
| 4. Dataset | one or more of the variants below | |
| 5. Export | ZIP or folder: mono 16-bit WAV, 32/40/48 kHz, README with training settings | |

### Dataset variants

1. **Clean voice from the track**: silences removed, light spectral noise gate, breaths kept / softened / removed,
   slices of 2–10 s cut at natural pauses, a 0–100 quality score for each slice (voiced share, noise, clipping, length).
   Bad slices are excluded automatically; every slice can be toggled in the UI.
2. **Text read in the singer's voice (TTS)**: one song gives only 1–3 minutes of voice, and RVC wants 10–30.
   So the singer's timbre is cloned with **Qwen3-TTS 1.7B fine-tuned to follow Russian stress marks**
   ([siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf)),
   and it reads a text. The default text is chapter 1 of *Harry Potter* in Russian (`harry.txt`); any text can be used.
   - **Stress marks**: [RUAccent](https://github.com/Den4ikAI/ruaccent) places stress and ё; homographs are resolved
     by context. Your stress dictionary and manual `+` marks (`з+амок` / `зам+ок`) win. Click any word in the UI to
     change its stress.
   - **Every block is checked after synthesis** (the approach of
     [Speech Studio](https://github.com/timoncool/speech-studio), extended):
     - Whisper large-v3-turbo transcribes the take and compares it with the text (word error rate within
       10 / 20 / 35 %);
     - **meter2** (wav2vec2 forced alignment + a stress head, about 98 % accurate) hears which syllable is actually
       stressed and compares it with the mark;
     - plus voice similarity to the reference, pauses, repetitions and length.
   - A failed block is re-generated with another seed (up to N attempts) and the best take is kept. Failed blocks
     stay out of the dataset unless you accept them by ear.
3. **Pitch-shifted copies**: clean slices shifted ±1–2 semitones by the WORLD vocoder. The formants (timbre) are
   preserved, which widens the pitch range.

### Recommended epochs

After the dataset is built, the **Dataset & training** tab shows the recommended **epochs** (with a sensible range),
the **batch size** for your GPU, the steps per epoch and in total, and how often to save. These numbers update
when you change which variants go into the export. The same numbers are written to the README inside the ZIP.

The recommendation uses the *effective* length: pitch-shifted copies count 30 %, because they are near-duplicates.

| effective minutes | < 3 | 3–6 | 6–12 | 12–25 | 25–45 | > 45 |
|---|---|---|---|---|---|---|
| epochs (range) | 400 (300–500) | 300 (250–400) | 250 (200–300) | 200 (150–250) | 150 (100–200) | 100 (60–150) |

## Install

**Windows 10/11**: double-click `install.bat`. It downloads portable Python 3.11, the right PyTorch build
(NVIDIA → CUDA 12.6, Intel Arc → XPU, otherwise CPU), the libraries and the neural networks (about 12 GB), and
creates a desktop shortcut. Nothing is installed system-wide.

```
install.bat            auto-detect
install.bat cpu        force the CPU build
set RVC_LITE=1         skip the TTS model at install time (it is downloaded on first use)
```

**Fully portable**: Python, PyTorch, libraries, models, the Hugging Face / PyTorch caches (`models/hf_home`,
`models/torch_home`) and projects all live in this folder. Nothing touches the registry or the user profile. The only
things created outside the folder are the desktop and Start-menu shortcuts (`set RVC_NO_SHORTCUT=1` skips them).
To move the studio, copy the folder; on a PC with a different GPU, run `install.bat` once to get the matching
PyTorch build. To uninstall, delete the folder.

**Linux / macOS**: `bash install.sh` (NVIDIA → CUDA, AMD → ROCm, Apple Silicon → MPS, otherwise CPU).

**Google Colab**: open `colab/RVCStudio_Colab.ipynb`, upload `dist/rvc-studio-app.zip` (built by
`python tools/make_dist.py`) or clone from git, then run cells 1–4. Optional settings in cell 2:

- `HF_TOKEN`: faster model downloads without rate limits.
- `NGROK_AUTHTOKEN`: a public ngrok link to the studio instead of the Colab proxy link.
- Store projects and models on Google Drive.

## Run

```
RVCStudio.bat                      web studio at http://127.0.0.1:8791/   (Linux/macOS: bash run.sh)
dataset.bat "song.wav" [--variants clean,tts,aug] [--blocks 60] [--text book.txt]    no browser
check-gpu.bat                      which device will be used and why
```

In the studio, drop a track on the left and press **Prepare dataset** (the defaults are fine). Then:

- listen to every stage in **Vocal**;
- untick bad slices in **1 · Clean voice**;
- fix stress by clicking words in **2 · TTS**;
- download the ZIP in **Dataset & training**, where the epoch count is shown.

`http://127.0.0.1:8791/#<project>/<tab>` deep-links to a project and a tab.

## Performance (measured)

3-minute song, 20 TTS blocks (~2.4 min of speech), all variants:

- **RTX 4060 Ti 16 GB**: about 4.5 minutes in total. Separation steps take ~25 s each, TTS with the checks ~7 s
  per block.
- **CPU (Ryzen 7 5800X)**: works, but slowly. Separation runs at about 2.7× the track length per step, TTS at
  ~5–7× real time. On CPU use the light 0.6B model and 10–20 blocks. The light model does not understand stress
  marks, so it gets plain text and the stress check re-generates its mistakes.
- TTS precision is chosen per GPU: bf16 on Ampere and newer (RTX 30/40/50, A100…), float32 on older cards
  (T4, GTX 10/16, RTX 20).

## Folders

```
app/            server.py (HTTP + API), make_dataset.py (CLI), selftest.py, web/ (UI)
app/studio/     separator, dataset (cleaning/slicing), textprep (text + stress), tts, verify, meter2, models, pipeline
models/         neural networks (downloaded by the installer; missing ones on demand)
projects/       results: vocal stages, dataset/clean|tts|aug, tts/takes (all takes), export/
python/         portable Python (Windows); .venv on Linux/macOS
```

## Limitations

- Effects baked into the voice (autotune, distortion, vocoder, chorus) are not removed by separation.
- TTS reproduces the singer's timbre in a speaking manner. Its bandwidth is 12 kHz (the TTS works at 24 kHz),
  which is fine for a 40k RVC model.
- The checks are measurements, not ears: listen to blocks marked red and to low-score slices.

## Licenses

See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Weights are downloaded from their original locations and
are not redistributed.

**Non-commercial use only** for two components:

- Sucial De-Echo is CC BY-NC-SA 4.0.
- The meter2 head was trained partly on Silero renders; its author calls it a research tool.

**Check before commercial use**: anvuew De-Reverb is GPL-3.0.

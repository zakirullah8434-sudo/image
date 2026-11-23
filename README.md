# Local Image Studio

A complete, **local, private, CPU-only** AI image generation web app built for a machine with
**8 GB RAM and no dedicated GPU**. Everything runs on your own PC — no cloud, no API keys, no
images leaving the machine.

| | |
|---|---|
| Model | Stable Diffusion 1.5 (pruned fp16 single-file `.safetensors`) |
| Runtime | PyTorch **CPU** build, `float32`, attention slicing + VAE slicing |
| Scheduler | `EulerAncestralDiscreteScheduler` — quality at 16–20 steps |
| Resolution | hard-clamped to **≤ 768 px** so 8 GB RAM never crashes |
| Backend | FastAPI + Uvicorn on `http://127.0.0.1:8000` |
| Frontend | one file, `frontend/index.html` (Tailwind via CDN, vanilla JS) |

---

## 1. Quick start (Windows)

Double-click **`start.bat`**.

That one script:

1. creates `venv\` (Python 3.10+ required, on `PATH`)
2. installs CPU-only PyTorch + all dependencies — **first run only**, ~2.5 GB
3. runs `backend/setup_model.py`, which downloads a ~2 GB checkpoint with a progress bar
   — **first run only**
4. starts the server and opens **http://127.0.0.1:8000** in your browser

Later launches take a few seconds: dependencies and the model are already cached.

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

---

## 2. How to use it

Open **http://127.0.0.1:8000**

1. **Prompt** — describe the image. Click the preset chips (*Photorealistic, Portrait,
   Cinematic, Anime, Fantasy, Sci-Fi, Product*) to append style keywords; click again to remove.
2. **Negative prompt** — pre-filled with a sensible default (low quality, deformed hands,
   watermark…). Edit or clear it freely.
3. **Aspect ratio** — `512×512 Square`, `512×768 Portrait`, `768×512 Landscape`.
4. **Steps** — slider `12…25`, default **18** (16–20 is the sweet spot for Euler Ancestral).
5. **CFG scale** — slider `1.0…12.0`, default **7.0**.
6. **Seed** — `−1` = random every time; click the 🎲 button for a random number, or paste a
   previous seed to reproduce an image exactly.
7. Press **Generate Image** (or `Ctrl + Enter`).

While it runs you get a **live elapsed timer** and a phase label. On CPU expect roughly:

| Size | Steps | Typical time |
|------|-------|--------------|
| 512×512 | 18 | ~1–2 min |
| 768×512 / 512×768 | 18 | ~2–4 min |
| 768×768 | 18 | ~3–6 min |

*(first generation is slower: the checkpoint is also being read from disk into RAM.)*

When it finishes you get:

- the image at full size
- **Download PNG** — one click, saves to your Downloads folder
- **Copy Seed** — paste it back into the seed box to recreate the image
- a metadata strip: time, seed, size, steps/CFG, peak RAM, model
- the image **auto-saved** into `outputs/` with a timestamp, and added to the
  **History gallery** — click any thumbnail to bring it back into the viewer

---

## 3. Free-RAM behaviour (important on 8 GB)

Stable Diffusion in `float32` needs roughly **4.6–5.3 GB** by itself. The app is built to stay
under a **5.5 GB peak** for the process, but your PC also runs Windows and everything else:

* **Under ~3.5 GB free** → generation still runs, but a toast warns you that closing apps will
  make it noticeably faster.
* **Under ~1.0 GB free** → the request is refused with HTTP `503` and a clear message instead
  of freezing/crashing your machine.
* Only **one generation runs at a time** (a second request gets `409`), so peak memory can
  never double.
* Resolution is clamped server-side to **768 px max** regardless of what the client sends.

> Tip: close browser tabs, game launchers and chat apps before long 768×768 runs.

Tune the thresholds with environment variables if you like:

```bat
set MIN_FREE_RAM_GB=1.0
set RAM_WARN_GB=3.5
set TORCH_NUM_THREADS=6
```

---

## 4. Adding your own models

Drop any **SD 1.5 single-file `.safetensors`** checkpoint into `models/` and press **↻ rescan**
in the UI (or restart). Every file appears in the model dropdown automatically; invalid or
half-downloaded files are greyed out as `⚠ invalid`.

Verified on install:

```
models/
├── Realistic_Vision_V6.0_NV_B1_fp16.safetensors   (auto-downloaded, ~2.0 GB)
├── dreamshaper_8.safetensors                      (fallback source)
└── your-model.safetensors                         (drop it here)
```

Manage downloads manually:

```bash
python backend/setup_model.py           # ensure a model exists
python backend/setup_model.py --list    # list what's installed
python backend/setup_model.py --check   # verify integrity (exact byte-length check)
python backend/setup_model.py --force   # re-download
```

If you already have a valid checkpoint in `models/`, setup **skips** the 2 GB download.

---

## 5. HTTP API

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/` | serves `frontend/index.html` |
| `GET` | `/api/models` | every `.safetensors` in `models/` (name, size, validity) |
| `POST` | `/api/generate` | run one generation → base64 PNG JSON |
| `GET` | `/api/gallery` | newest-first list of images in `outputs/` |
| `GET` | `/api/health` | liveness, free RAM, loaded model |
| `GET` | `/outputs/{name}.png` | serves a saved image |

`POST /api/generate` body:

```json
{
  "prompt": "a cinematic portrait of ...",
  "negative_prompt": "low quality, ...",
  "width": 512, "height": 768,
  "steps": 18, "guidance_scale": 7.0,
  "seed": -1,
  "model_name": "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
}
```

Response includes `image_base64`, `seed`, `generation_time_seconds`, `model_load_seconds`,
`peak_ram_gb`, `ram_warning`, `filename`, plus the clamped `width`/`height`/`steps`.

Interactive docs are at **http://127.0.0.1:8000/docs**.

---

## 6. Project structure

```
local-image-studio/
├── backend/
│   ├── main.py             # FastAPI server + memory-safe pipeline
│   ├── setup_model.py      # model auto-downloader / integrity verifier
│   └── requirements.txt    # CPU-only dependency pins
├── frontend/
│   └── index.html          # the whole studio UI (single file)
├── models/                 # .safetensors checkpoints (auto-created)
├── outputs/                # auto-saved timestamped PNGs
├── venv/                   # virtual environment (auto-created)
├── start.bat               # Windows one-click launch
└── start.sh                # Linux / macOS launch
```

### Manual install (instead of `start.bat`)

```bash
python -m venv venv
venv\Scripts\pip install -r backend\requirements.txt   # CPU torch wheel
venv\Scripts\python backend\setup_model.py
venv\Scripts\python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| *"No model found in models/"* | run `python backend/setup_model.py`, or drop a `.safetensors` into `models/` |
| *"Only X GB of RAM is free…" (503)* | close other apps; the guard prevents a crash |
| Very slow generation | close apps, use 512×512 and 12–16 steps; set `TORCH_NUM_THREADS` to your physical core count |
| Port already in use | `set PORT=8001` before `start.bat`, or stop the other process |
| Download interrupted | `python backend/setup_model.py` resumes/ retries automatically |
| Want to re-download the model | delete the file in `models\` then run `setup_model.py --force` |
| Need more page file | System → Advanced → Performance → Virtual memory (the app never relies on it, but it helps on 8 GB) |

**Requirements:** Python 3.10–3.14 (64-bit), ~5 GB free disk for deps + model, internet on
first run only.

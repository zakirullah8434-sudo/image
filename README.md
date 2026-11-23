# Local Image Studio

A complete, **local, private, CPU-only** AI image generation web app built for a machine with
**8 GB RAM and no dedicated GPU**. Everything runs on your own PC — no cloud, no API keys, no
images leaving the machine.

| | |
|---|---|
| Model | Stable Diffusion 1.5 (pruned fp16 single-file `.safetensors`) |
| Runtime | PyTorch **CPU** build, `float32`, VAE slicing, PyTorch SDPA attention |
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
4. **Steps** — slider `12…25`, default **12** (fastest). Raise toward 20 for extra detail at
   roughly proportional cost.
5. **CFG scale** — slider `1.0…12.0`, default **7.0**. **Set it to `1.0` to almost halve the
   render time** — the model then skips the second (negative-prompt) pass — at the cost of how
   strictly the prompt is followed.
6. **Seed** — `−1` = random every time; click the 🎲 button for a random number, or paste a
   previous seed to reproduce an image exactly.
7. Press **Generate Image** (or `Ctrl + Enter`).

While it runs you get a **live elapsed timer** and a phase label. Measured on a 4-core
i5-3470 with ~1.5 GB RAM free (page file on an SSD):

| Size | Steps | Typical time |
|------|-------|--------------|
| 512×512 | 12 | ~7–8 min |
| 512×512 | 20 | ~12 min |
| 768×512 / 512×768 | 12 | ~12–16 min |
| 768×768 | 12 | ~18–25 min |

**This machine is memory-bound, not just CPU-bound**: the `float32` model wants ~4.3 GB and
only ~1.5 GB is free, so Windows streams weights back from the page file on every step.
Freeing RAM (close tabs/launchers) is the single biggest speed-up available, followed by
CFG `1.0`, then fewer steps.

*(First generation is slower still: the checkpoint is read from disk into RAM, ~1.5 min.)*

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

**Attention slicing** is deliberately **off by default**: PyTorch ≥ 2.0 already runs memory-efficient
SDPA attention, and slicing on top of it makes the CPU walk attention one row at a time — measured
**51 s/step instead of ~8**. Only turn it on if you genuinely run out of RAM:

```bat
set ATTENTION_SLICING=1     # slower but leaner; default 0 (off)
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
  "steps": 12, "guidance_scale": 7.0,
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
├── models/                 # .safetensors checkpoints (auto-created, never committed)
├── outputs/                # auto-saved timestamped PNGs (never committed)
├── venv/                   # virtual environment (auto-created)
├── start.bat               # Windows one-click launch
├── start.sh                # Linux / macOS launch
├── tunnel.bat              # public https:// tunnel for the backend (Vercel/phone use)
├── vercel.json             # static-host config: serves frontend/ only
├── .gitignore              # keeps venv/ 2 GB model/outputs out of git
└── .vercelignore           # keeps Vercel's upload limited to frontend/
```

### Manual install (instead of `start.bat`)

```bash
python -m venv venv
venv\Scripts\pip install -r backend\requirements.txt   # CPU torch wheel
venv\Scripts\python backend\setup_model.py
venv\Scripts\python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

---

## 7. Use it from your phone (host the UI on Vercel)

The same `frontend/index.html` works two ways:

* **On your PC** — opened via `start.bat`, the page talks to the server it was served from
  (no configuration).
* **Hosted (Vercel)** — the page needs the address of *your PC's* backend, which you expose
  with a free public tunnel. Images are still generated on your own machine; nothing goes to a
  third-party AI service.

> Vercel is **static hosting only**: it cannot run PyTorch or hold your 2 GB model. That's why
> `vercel.json` points at `frontend/` and the backend stays on your PC.

### Step 1 — Vercel

1. Push this repo to GitHub (already done for this project).
2. Go to **vercel.com → Add New → Project → Import** your repository.
3. Keep the defaults — `vercel.json` already sets the output directory to `frontend`.
4. Deploy. You get an address like `https://image-xxxx.vercel.app`.

### Step 2 — expose your PC

1. Run **`start.bat`** and leave it open (backend on `127.0.0.1:8000`).
2. Run **`tunnel.bat`** in a second window. It downloads cloudflared once (~40 MB), then
   prints a public address:

   ```
   https://xxxx-xxxx.trycloudflare.com
   ```

3. Leave that window open — closing it stops the tunnel. The address changes each run.

### Step 3 — connect on any device

1. Open your Vercel URL on PC **or phone**.
2. Click the **`local`** button in the top bar → paste the `https://….trycloudflare.com`
   address → **Connect**.
3. Status turns green (*Server online*), the model list and gallery load, and Generate works.
   The choice is remembered in that browser (`localStorage`), so you only do this once per
   device.

Shortcut: you can also open `https://your-app.vercel.app/?api=https://xxxx.trycloudflare.com`
to connect without clicking through the modal.

**Requirements:** `start.bat` **and** `tunnel.bat` must stay running on your PC, and the PC
must stay awake. Tunneling is a plain HTTPS proxy — your prompts and images travel only
between your device and your own machine.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| *"No model found in models/"* | run `python backend/setup_model.py`, or drop a `.safetensors` into `models/` |
| *"Only X GB of RAM is free…" (503)* | close other apps; the guard prevents a crash |
| Very slow generation | close apps, use 512×512 and 12–16 steps; set `CFG 1.0`; set `TORCH_NUM_THREADS` to your physical core count |
| Port already in use | `set PORT=8001` before `start.bat`, or stop the other process |
| Download interrupted | `python backend/setup_model.py` resumes/ retries automatically |
| Want to re-download the model | delete the file in `models\` then run `setup_model.py --force` |
| Need more page file | System → Advanced → Performance → Virtual memory (the app never relies on it, but it helps on 8 GB) |
| Hosted page says **Offline** / modal about a backend | expected on Vercel until you connect: run `start.bat` + `tunnel.bat` on the PC and paste the tunnel URL via the **local** button |
| *"did not answer with JSON"* / tunnel errors | `tunnel.bat` window was closed or the PC slept — rerun it and reconnect with the **new** address |
| `409 Conflict` on Generate | one generation is already running — wait for it (a second one would double peak RAM) |
| Images 404 on the hosted page | the stored tunnel address is stale — reconnect via the **local** button with the current `trycloudflare.com` URL |

**Requirements:** Python 3.10–3.14 (64-bit), ~5 GB free disk for deps + model, internet on
first run only.

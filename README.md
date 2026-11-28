# Local Image Studio

A complete, **local, private, CPU-only** AI image generation web app built for a machine with
**8 GB RAM and no dedicated GPU** — plus an **Online mode** that creates images through a
hosted service straight from the browser, so it works even when the PC is off.

| | |
|---|---|
| Model | Stable Diffusion 1.5 (pruned fp16 single-file `.safetensors`) |
| Runtime | PyTorch **CPU** build, `float32`, VAE slicing, PyTorch SDPA attention |
| Scheduler | `EulerAncestralDiscreteScheduler` — quality at 16–20 steps |
| Resolution | hard-clamped to **≤ 768 px** so 8 GB RAM never crashes |
| Backend | FastAPI + Uvicorn on `http://127.0.0.1:8000` |
| Frontend | one file, `frontend/index.html` (Tailwind via CDN, vanilla JS) |
| Modes | **Local** (this PC, private, img2img) and **Online** (hosted, PC not needed) |

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

Open **http://127.0.0.1:8000** (or the hosted Vercel URL).

**First you pick a mode** — a chooser appears every time the page opens, and the
**pill in the top bar** switches any time:

| Mode | What it does |
|------|--------------|
| 🌐 **Online — hosted** | Images are created by a hosted AI service called directly from your browser. **Your PC does not need to be on**, works on any phone/laptop, no setup, **and you can upload a photo to edit it (img2img)**. Results look photorealistic by default (style words like *anime/cartoon* switch that off). Your prompt is sent to that third party. |
| 🖥️ **Local — on this PC** | Stable Diffusion runs on your own computer — 100% private, offline, supports image editing (img2img) and your own model files. Needs `start.bat` (and `tunnel.bat` for phones). |

The steps below describe **Local mode**:

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

**Editing an existing image (img2img):** click **Attach an image to edit** under the
prompt (or **Edit this image** on any result / gallery image) and pick a PNG, JPG or WebP.
The image appears as a thumbnail with an **Edit strength** slider — `0.10` keeps almost
everything, `0.95` almost reinvents the picture — and Generate starts from your image
instead of noise. The upload is downscaled to ≤1024 px inside the browser, sent only to
your own backend, and never stored anywhere.

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
- **Edit this image** — attach the result again for another round of edits
- the image **auto-saved** into `outputs/` with a timestamp, and added to the
  **History gallery** — click any thumbnail to bring it back into the viewer

---

## 3. Free-RAM behaviour (important on 8 GB)

Stable Diffusion in `float32` needs roughly **4.6–5.3 GB** by itself. The app is built to stay
under a **5.5 GB peak** for the process, but your PC also runs Windows and everything else:

* **Under ~3.5 GB free** → generation still runs, but a toast warns you that closing apps will
  make it noticeably faster.
* **Under ~0.4 GB free** → the request is refused with HTTP `503` and a clear message instead
  of freezing/crashing your machine. This machine normally idles around **1 GB free** and lets
  Windows page cold model weights to disk, so the floor is deliberately low — it only trips
  when there is essentially no commit headroom left at all.
* Only **one generation runs at a time** (a second request gets `409`), so peak memory can
  never double.
* Resolution is clamped server-side to **768 px max** regardless of what the client sends.

> Tip: close browser tabs, game launchers and chat apps before long 768×768 runs.

Tune the thresholds with environment variables if you like:

```bat
set MIN_FREE_RAM_GB=0.4
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
| `POST` | `/api/generate` | run one generation → base64 PNG JSON (txt2img, or img2img when `image_b64` is set) |
| `GET` | `/api/job?since=<epoch>` | current/last job — poll this if your generate POST was cut mid-run |
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
  "model_name": "Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
  "image_b64": "data:image/jpeg;base64,...",
  "strength": 0.55
}
```

`image_b64` + `strength` are optional — with them the run becomes an **img2img edit**
(the image is the starting point, `strength` 0.05–0.95 = how far the prompt may change
it). Response includes `image_base64`, `seed`, `generation_time_seconds`, `model_load_seconds`,
`peak_ram_gb`, `ram_warning`, `filename`, `mode` (`txt2img`/`img2img`) + `strength`, plus the
clamped `width`/`height`/`steps`.

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
│   ├── index.html          # the whole studio UI (single file)
│   ├── 404.html            # sends unknown URLs back to the app
│   └── backend.json        # current tunnel address (written by tunnel.bat)
├── models/                 # .safetensors checkpoints (auto-created, never committed)
├── outputs/                # auto-saved timestamped PNGs (never committed)
├── venv/                   # virtual environment (auto-created)
├── start.bat               # Windows one-click launch
├── start.sh                # Linux / macOS launch
├── autorun.bat             # auto-start watchdog: backend + tunnel, restarts every 30 s
├── autorun.vbs             #   └─ hidden launcher (no window at sign-in)
├── install-autostart.bat   # one-time setup: puts autorun into Windows Startup
├── install-autostart.ps1   #   └─ creates the Startup shortcut
├── stop.bat / stop.ps1     # stops backend + tunnel + watchdog
├── tunnel.bat              # public https:// tunnel for the backend (Vercel/phone use)
├── tunnel.ps1              #   └─ publishes the tunnel address to the site by itself
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
4. **Double-check the project settings** (this is what makes or breaks the deploy):
   * **Settings → General → Root Directory** must be **empty** (repo root). If it says
     `backend`, Vercel looks for `backend/frontend`, finds nothing and ships an **empty
     site — every URL returns 404**.
   * **Framework Preset** must be **Other** (not `FastAPI`/`Python` — there is nothing to
     build here). `vercel.json` in the repo already pins `framework: null`.
   * **Settings → Deployment Protection → Require Log In** must be **off**, otherwise every
     visitor (including your phone) gets a Vercel login page instead of the app.
5. Deploy. You get an address like `https://skyhoniximage.vercel.app`.

### Step 2 — expose your PC

1. Run **`start.bat`** and leave it open (backend on `127.0.0.1:8000`).
2. Run **`tunnel.bat`** in a second window. It downloads cloudflared once (~40 MB), starts a
   public address like

   ```
   https://xxxx-xxxx.trycloudflare.com
   ```

   and **publishes that address to the website by itself** — it writes `frontend/backend.json`,
   commits and pushes, and the hosted page picks the new address up (usually within seconds via
   the raw GitHub copy, at worst after the ~1-minute Vercel rebuild). Nothing to copy, nothing
   to paste.
3. Leave that window open — closing it stops the tunnel. The address changes each run, which is
   fine: every run is published the same automatic way.

### Step 3 — connect on any device

1. Open your Vercel URL on PC **or phone**.
2. Wait for it: the page probes your PC every few seconds and the status turns green
   (*Server online*) by itself — the model list and gallery load and Generate works.
   It reconnects automatically after `tunnel.bat` restarts with a brand-new address.
3. The working address is remembered in that browser (`localStorage`), so later visits
   reconnect even faster.

The **`local`** button in the top bar remains as a manual override (for example to point at a
LAN IP), and opening `https://your-app.vercel.app/?api=https://xxxx.trycloudflare.com` still
forces a specific backend.

**Requirements:** `start.bat` **and** `tunnel.bat` must stay running on your PC, and the PC
must stay awake. Tunneling is a plain HTTPS proxy — your prompts and images travel only
between your device and your own machine.

---

## 8. Run it automatically with Windows (recommended)

After **one** double-click, the backend and the public tunnel start with Windows, restart
themselves if they ever die, and the hosted page connects by itself — on PC and phone.

### One-time setup

1. Double-click **`install-autostart.bat`**.
2. Done — it creates a hidden shortcut in your Startup folder and starts the watchdog
   right away (no sign-out needed).

What runs from now on, **every time you sign in to Windows**:

| Piece | What it does |
|---|---|
| `autorun.vbs` (hidden) | starts `autorun.bat` with **no window** |
| `autorun.bat` (watchdog) | ① starts the backend if port 8000 is dead → ② waits for `/api/health` (~60 s while the model server warms up) → ③ starts the tunnel if cloudflared isn't running → ④ **re-checks every 30 s and restarts whatever died** |
| `tunnel.ps1` (auto mode) | publishes the new `https://….trycloudflare.com` address to the site by itself (same as always), without waiting for a key press |

Logs go to `%TEMP%\lis-autorun.log`. **Stop everything** with **`stop.bat`**
(the watchdog returns at the next sign-in). **Start manually again** with `start.bat`
and/or `tunnel.bat` whenever you like — they coexist with the watchdog.

> The **first** start after boot takes about a minute (PyTorch loads the model); the
> status pill on the hosted page turns green by itself. If the PC sleeps, the tunnel
> drops — keep the PC awake (Settings → Power) or just wake it; the page reconnects
> automatically.

### Manual step-by-step (no auto-start)

1. Double-click **`start.bat`** → backend runs on `http://127.0.0.1:8000` (leave the window open).
2. Double-click **`tunnel.bat`** in a second window → public address is published to the
   site automatically (leave that window open too).
3. Open **https://skyhoniximage.vercel.app** on PC or phone → the status pill turns
   *Server online* by itself within about a minute. Nothing to paste.
4. To stop: close both windows (or run `stop.bat`).

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| *"No model found in models/"* | run `python backend/setup_model.py`, or drop a `.safetensors` into `models/` |
| *"Only X GB of RAM is free…" (503)* | close other apps; the guard only trips below **0.4 GB** free (this PC normally idles at ~1 GB and pages normally) |
| *Phone: "Connection dropped — your PC keeps generating…"* | normal — Cloudflare cuts long requests at ~100 s, so the page polls `/api/job` and shows the image when it finishes |
| *Phone: "Cannot reach the PC/backend…"* | the PC is off or asleep, or the tunnel died — run `install-autostart.bat` **once** so both start with Windows; otherwise run `start.bat` + `tunnel.bat` and keep them open. The page re-probes by itself every ~15 s |
| Very slow generation | close apps, use 512×512 and 12–16 steps; set `CFG 1.0`; set `TORCH_NUM_THREADS` to your physical core count |
| Port already in use | `set PORT=8001` before `start.bat`, or stop the other process |
| Download interrupted | `python backend/setup_model.py` resumes/ retries automatically |
| Want to re-download the model | delete the file in `models\` then run `setup_model.py --force` |
| Need more page file | System → Advanced → Performance → Virtual memory (the app never relies on it, but it helps on 8 GB) |
| Hosted page says **Offline** / modal about a backend | your PC is off or only `start.bat`/`tunnel.bat` are missing — run `install-autostart.bat` once and it never happens again; otherwise open `start.bat` + `tunnel.bat` and leave both open. The page connects **by itself** within a minute (it re-checks every few seconds) |
| *"did not answer with JSON"* / tunnel errors | `tunnel.bat` window was closed or the PC slept — rerun it; the page picks up the **new** address automatically, nothing to paste |
| `409 Conflict` on Generate | another image is really still generating — the page now detects that and waits for it automatically; if nothing is running, just press Generate again |
| Images 404 on the hosted page | the saved tunnel address is stale — rerun `tunnel.bat`; the page refreshes its address automatically (or override it with the **local** button) |
| Unknown/deep URL shows a 404 | `frontend/404.html` redirects unknown paths back to the app — if you get Vercel's plain 404 instead, that file was deleted from `frontend/` |
| Hosted URL returns **404 on every path** (incl. `/`) | Vercel **Root Directory** is set to something (e.g. `backend`) — clear it in Settings → General, and confirm Framework Preset = **Other**. Then redeploy |
| Hosted URL shows a **Vercel "Log in" page** | Deployment Protection → **Require Log In** is on — turn it off (Settings → Deployment Protection) |
| `skyhoniximage.vercel.app` 404 but the deployment looks Ready | the deployment predates the settings fix — push a commit (or click **Redeploy**) so Vercel rebuilds with the correct root |

**Requirements:** Python 3.10–3.14 (64-bit), ~5 GB free disk for deps + model, internet on
first run only.

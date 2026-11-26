"""
Local Image Studio - memory-safe CPU inference server (FastAPI).

Designed strictly for an 8 GB RAM / no-GPU machine:

* one pipeline in memory at a time, loaded lazily on first generation
* float32 weights (spec) + attention slicing + VAE slicing to cap activations
* garbage collection before and after every generation
* resolution hard-clamped to 768 px, generations serialised behind a lock
* a free-RAM guard so the process degrades with a clear message instead of
  crashing the machine when other applications are using the memory
* every PNG is written to ``outputs/`` with a timestamp

Run:  uvicorn main:app --host 127.0.0.1 --port 8000   (from backend/)
"""

from __future__ import annotations

import base64
import gc
import io
import os
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from setup_model import FRONTEND_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_dirs, list_models, verify_model

# --------------------------------------------------------------------------- #
# Tunables - everything here exists to keep peak RSS under ~5.5 GB
# --------------------------------------------------------------------------- #
MAX_DIM = 768          # hard ceiling for width/height (8 GB RAM safety)
MIN_DIM = 256
MAX_STEPS = 50
MIN_STEPS = 1
MAX_GUIDANCE = 20.0
MIN_GUIDANCE = 1.0

# Refuse to start a generation only in a genuine emergency (essentially no
# commit headroom at all). This machine normally runs with ~1 GB free and
# leans on the page file - the README documents paging as its normal mode -
# so the floor is deliberately low: Windows pages cold model weights out
# while the CPU grinds. Below the floor even opening a window would fail.
# Override with the MIN_FREE_RAM_GB environment variable.
MIN_FREE_RAM_GB = float(os.environ.get("MIN_FREE_RAM_GB", "0.4"))

# Softer threshold: below this we still generate (8 GB machines rarely have
# more free), but we tell the user that closing apps will make it faster.
RAM_WARN_GB = float(os.environ.get("RAM_WARN_GB", "3.5"))

GALLERY_LIMIT = 120
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.png$")

# Uploaded-image guards (img2img editing). The UI downscales to <=1024 px on
# the longest side before sending, so honest uploads are far below these.
MAX_IMAGE_BYTES = 25 * 1024 * 1024   # refuse absurd payloads early
MAX_IMAGE_PIXELS = 4096 * 4096       # decompression-bomb guard

# torch/diffusers configuration
torch.set_grad_enabled(False)
if os.environ.get("TORCH_NUM_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))

app = FastAPI(
    title="Local Image Studio",
    description="Local CPU-only Stable Diffusion 1.5 image generation",
    version="1.0.0",
)

# One build serves two ways: directly from this process (same origin) and from
# a static host such as Vercel, which reaches us through a public tunnel. That
# second case is a cross-origin browser request, so allow it. No cookies or
# credentials are ever sent, and every mutating route is a plain JSON POST.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
_pipe = None                      # loaded StableDiffusionPipeline (or None)
_pipe_model: str | None = None    # file name currently loaded
_img2img = None                   # img2img view over _pipe (shares the weights)
_gen_lock = threading.Lock()      # serialise generation - never two at once
_model_lock = threading.Lock()    # serialise model (re)loading

# Current/last job - lets a client whose POST was cut mid-generation (Cloudflare
# drops proxied requests after ~100 s while a CPU run takes minutes) poll
# /api/job for the finished payload instead of failing. See api_job().
_job_lock = threading.Lock()
_job: dict = {
    "running": False,
    "started_at": 0.0,
    "result": None,
    "result_at": 0.0,
    "error": None,
    "error_at": 0.0,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clamp_dim(value: int) -> int:
    """Clamp to [MIN_DIM, MAX_DIM] and snap to a multiple of 8 (SD latent size)."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 512
    v = max(MIN_DIM, min(MAX_DIM, v))
    return v - (v % 8)


def _clamp_int(value: int, lo: int, hi: int, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clamp_float(value: float, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _process_ram_gb() -> tuple[float, float]:
    """Return ``(current_gb, peak_gb)`` resident memory of this process."""
    info = psutil.Process().memory_info()
    current = info.rss / 1024**3
    peak = getattr(info, "peak", info.rss) / 1024**3
    return current, peak


def _system_free_gb() -> float:
    return psutil.virtual_memory().available / 1024**3


def _free_ram_guard() -> None:
    """Fail fast in a genuine emergency instead of thrashing the page file."""
    free = _system_free_gb()
    if free < MIN_FREE_RAM_GB:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Only {free:.1f} GB of RAM is free right now - image generation cannot "
                "start safely (float32 Stable Diffusion needs several GB). "
                "Close other applications (browser tabs, games, launchers) and try again."
            ),
        )
    return None


def _ram_note() -> str | None:
    """Human-readable advice when free RAM is on the low side."""
    free = _system_free_gb()
    if free < RAM_WARN_GB:
        return (
            f"Only {free:.1f} GB RAM free - generation will work but stays faster if you "
            f"close apps first (ideal: {RAM_WARN_GB:.1f}+ GB free)."
        )
    return None


def _resolve_model_name(model_name: str | None) -> Path:
    """Validate a requested model file and return its path."""
    available = list_models()
    if not available:
        raise HTTPException(
            status_code=400,
            detail=(
                "No model found in models/. Run `python backend/setup_model.py` to "
                "download a checkpoint, or drop any SD 1.5 .safetensors file in there."
            ),
        )

    if model_name:
        # Only allow plain file names that really exist (path-traversal safe).
        if Path(model_name).name != model_name:
            raise HTTPException(status_code=400, detail="Invalid model name.")
        match = [p for p in available if p.name == model_name]
        if not match:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not found in models/.",
            )
        return match[0]

    # No preference -> first verified model, else simply the first one.
    for p in available:
        ok, _ = verify_model(p)
        if ok:
            return p
    return available[0]


def load_pipeline(model_path: Path):
    """Load (or swap) the single-file checkpoint. Called under ``_model_lock``."""
    global _pipe, _pipe_model, _img2img

    if _pipe is not None and _pipe_model == model_path.name:
        return _pipe

    # Drop the previous pipeline before loading a new one: only ever one copy.
    if _pipe is not None:
        _pipe.to("cpu")
        del _pipe
        _pipe = None
        _pipe_model = None
        _img2img = None              # its components went away with _pipe
        gc.collect()

    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline

    gc.collect()
    t0 = time.perf_counter()

    # --- Zero-filter pipeline (project spec) -------------------------------- #
    pipe = StableDiffusionPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # --- Fast CPU scheduler: quality at 16-20 steps ------------------------- #
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # --- 8 GB RAM optimisations -------------------------------------------- #
    # PyTorch >= 2.0 already uses SDPA (AttnProcessor2_0), which is memory
    # efficient on its own. Forcing sliced attention ON TOP of SDPA makes the
    # CPU walk attention one row at a time: measured 51 s/step instead of ~8.
    # diffusers documents this exact footgun, so slicing stays opt-in for
    # machines that genuinely run out of RAM (ATTENTION_SLICING=1).
    if os.environ.get("ATTENTION_SLICING", "0").strip().lower() in ("1", "true", "yes", "on"):
        pipe.enable_attention_slicing()
        print("[backend] attention slicing ENABLED (ATTENTION_SLICING=1) - slower but leaner")
    # VAE slicing costs almost nothing and keeps the decode flat on 8 GB.
    # diffusers >= 0.30 moved it off the pipeline onto the VAE component.
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)

    pipe.to("cpu")
    gc.collect()

    _pipe = pipe
    _pipe_model = model_path.name
    print(f"[backend] loaded {model_path.name} in {time.perf_counter() - t0:.1f}s "
          f"(RSS {_process_ram_gb()[0]:.2f} GB)")
    return pipe


def _next_output_path(seed: int, width: int, height: int) -> Path:
    """Build a collision-free, timestamped path inside ``outputs/``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_seed{seed}_{width}x{height}"
    path = OUTPUTS_DIR / f"{base}.png"
    counter = 1
    while path.exists():
        path = OUTPUTS_DIR / f"{base}_{counter}.png"
        counter += 1
    return path


def _decode_input_image(raw: str) -> Image.Image:
    """Decode a ``data:...;base64,...`` URL or raw base64 string into RGB pixels."""
    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="image_b64 is empty.")
    payload = raw.split(",", 1)[1] if raw.startswith("data:") and "," in raw else raw
    if len(payload) > MAX_IMAGE_BYTES * 2:   # cheap pre-check before decoding
        raise HTTPException(status_code=400, detail="Image is too large (25 MB max).")
    try:
        blob = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail="image_b64 is not valid base64."
        ) from exc
    if len(blob) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image is too large (25 MB max).")
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()                     # force decode now; catches truncated files
    except Exception as exc:            # noqa: BLE001 - any PIL failure = bad upload
        raise HTTPException(
            status_code=400,
            detail="Could not decode that image - use PNG, JPEG or WebP.",
        ) from exc
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400, detail="Image resolution is too high (max 4096x4096)."
        )
    return img.convert("RGB")


def _fit_image(img: Image.Image, box_w: int, box_h: int) -> tuple[Image.Image, int, int]:
    """Scale into the requested box, keep aspect ratio, snap to /8 for SD.

    Returns ``(resized_image, width, height)`` - both clamped to <= MAX_DIM.
    """
    box_w = _clamp_dim(box_w)
    box_h = _clamp_dim(box_h)
    scale = min(box_w / img.width, box_h / img.height)
    w = int(img.width * scale)
    h = int(img.height * scale)
    w = max(64, min(MAX_DIM, w - (w % 8)))
    h = max(64, min(MAX_DIM, h - (h % 8)))
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    return img, w, h


# --------------------------------------------------------------------------- #
# Request schema
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    width: int = 512
    height: int = 512
    steps: int = 12
    guidance_scale: float = 7.0
    seed: int = -1
    model_name: str = ""
    # --- img2img editing (optional) ------------------------------------- #
    # When set, the image becomes the starting point and ``strength`` decides
    # how far the prompt may transform it (0.05 = keep almost everything,
    # 0.95 = almost a fresh generation that still follows the composition).
    image_b64: str = ""              # data URL or raw base64 of the upload
    strength: float = 0.55


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-file studio UI."""
    page = FRONTEND_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(page, media_type="text/html; charset=utf-8")


@app.get("/api/models")
def api_models() -> list[dict]:
    """List every ``.safetensors`` file present in ``models/``."""
    result = []
    for path in list_models():
        ok, reason = verify_model(path)
        result.append(
            {
                "name": path.name,
                "size_gb": round(path.stat().st_size / 1024**3, 2),
                "valid": ok,
                "detail": reason,
                "loaded": path.name == _pipe_model,
            }
        )
    return result


@app.get("/api/gallery")
def api_gallery() -> list[dict]:
    """Newest-first list of images already saved in ``outputs/``."""
    if not OUTPUTS_DIR.is_dir():
        return []
    files = [p for p in OUTPUTS_DIR.iterdir() if p.is_file() and p.suffix == ".png"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files[:GALLERY_LIMIT]:
        st = p.stat()
        out.append(
            {
                "name": p.name,
                "url": f"/outputs/{p.name}",
                "created": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "size_kb": round(st.st_size / 1024, 1),
            }
        )
    return out


@app.get("/outputs/{name}", include_in_schema=False)
def get_output(name: str) -> FileResponse:
    """Serve one generated PNG (guarded against path traversal)."""
    if not SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    path = OUTPUTS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/generate")
def api_generate(req: GenerateRequest) -> dict:
    """Run one CPU generation and return the PNG as base64 JSON.

    Without ``image_b64`` this is plain txt2img; with it, img2img editing.
    """
    global _pipe, _pipe_model, _img2img

    # ---- validate + clamp (768 px ceiling so 8 GB RAM never crashes) ------- #
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    width = _clamp_dim(req.width)
    height = _clamp_dim(req.height)
    steps = _clamp_int(req.steps, MIN_STEPS, MAX_STEPS, 12)
    guidance = _clamp_float(req.guidance_scale, MIN_GUIDANCE, MAX_GUIDANCE, 7.0)
    negative = (req.negative_prompt or "").strip()
    model_path = _resolve_model_name((req.model_name or "").strip())

    # ---- optional input image (decoded before the lock: fails fast) -------- #
    init_image = None
    strength_val = None
    if (req.image_b64 or "").strip():
        init_image, width, height = _fit_image(_decode_input_image(req.image_b64), width, height)
        strength_val = _clamp_float(req.strength, 0.05, 0.95, 0.55)

    # ---- never two generations at once (would double peak RAM) ------------ #
    if not _gen_lock.acquire(blocking=False):
        if init_image is not None:
            init_image.close()
        raise HTTPException(
            status_code=409,
            detail="A generation is already running - please wait for it to finish.",
        )

    load_seconds = 0.0
    try:
        with _job_lock:
            _job["running"] = True
            _job["started_at"] = time.time()

        # ---- 8 GB RAM hygiene -------------------------------------------- #
        gc.collect()
        _free_ram_guard()
        ram_warning = _ram_note()

        with _model_lock:
            already_loaded = _pipe is not None and _pipe_model == model_path.name
            t_load = time.perf_counter()
            pipe = load_pipeline(model_path)
            load_seconds = 0.0 if already_loaded else time.perf_counter() - t_load

        # ---- img2img wrapper: shares _pipe's weights (no extra model RAM) -- #
        if init_image is not None and _img2img is None:
            from diffusers import StableDiffusionImg2ImgPipeline

            try:
                _img2img = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
            except Exception:            # noqa: BLE001 - older/newer API shapes
                _img2img = StableDiffusionImg2ImgPipeline(**pipe.components)
            _img2img.set_progress_bar_config(disable=True)

        # ---- seed ---------------------------------------------------------- #
        seed = int(req.seed)
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "little")  # 0 .. 4294967295
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ---- inference ----------------------------------------------------- #
        gc.collect()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if init_image is not None:
                result = _img2img(
                    prompt=prompt,
                    negative_prompt=negative or None,
                    image=init_image,
                    strength=strength_val,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    output_type="pil",
                )
            else:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative or None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    output_type="pil",
                )
        elapsed = time.perf_counter() - t0

        if init_image is not None:
            init_image.close()
            init_image = None
        image = result.images[0]

        # ---- save with a timestamp ---------------------------------------- #
        out_path = _next_output_path(seed, width, height)
        image.save(out_path, format="PNG", optimize=False)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        del buffer, image, result

        gc.collect()
        current_ram, peak_ram = _process_ram_gb()

        response = {
            "image_base64": image_b64,
            "mime": "image/png",
            "filename": out_path.name,
            "url": f"/outputs/{out_path.name}",
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "mode": "img2img" if strength_val is not None else "txt2img",
            "strength": round(strength_val, 2) if strength_val is not None else None,
            "model_name": model_path.name,
            "generation_time_seconds": round(elapsed, 2),
            "model_load_seconds": round(load_seconds, 2),
            "total_time_seconds": round(elapsed + load_seconds, 2),
            "ram_used_gb": round(current_ram, 2),
            "peak_ram_gb": round(peak_ram, 2),
            "system_free_ram_gb": round(_system_free_gb(), 2),
            "ram_warning": ram_warning,
        }
        with _job_lock:
            _job.update(running=False, result=response, result_at=time.time())
        return response

    except HTTPException as exc:
        # Record it too: a phone polling /api/job must see why it failed.
        with _job_lock:
            _job.update(
                running=False,
                error={"message": str(exc.detail)},
                error_at=time.time(),
            )
        raise
    except Exception as exc:  # noqa: BLE001 - keep the server alive
        gc.collect()
        traceback.print_exc()
        # CPU OOM surfaces as MemoryError or as torch's allocator RuntimeError.
        low_mem = isinstance(exc, MemoryError) or any(
            key in str(exc).lower()
            for key in ("not enough memory", "defaultcpuallocator", "out of memory")
        )
        detail = (
            "Ran out of memory. Close other applications, lower the resolution "
            "to 512x512, reduce steps, and try again."
            if low_mem
            else f"Generation failed: {type(exc).__name__}: {exc}"
        )
        with _job_lock:
            _job.update(running=False, error={"message": detail}, error_at=time.time())
        raise HTTPException(
            status_code=507 if low_mem else 500, detail=detail
        ) from exc
    finally:
        with _job_lock:
            _job["running"] = False
        _gen_lock.release()


@app.get("/api/job")
def api_job(since: float = 0.0) -> dict:
    """Polling endpoint for clients whose POST was cut mid-generation.

    Cloudflare's edge drops proxied requests after ~100 s while a CPU
    generation takes minutes. The client passes the epoch time just before
    it sent the POST (``since``) and gets back anything that finished after
    that moment - the full payload, ready to render.
    """
    now = time.time()
    with _job_lock:
        started = _job["started_at"]
        return {
            "running": _job["running"],
            "started_at": started,
            "running_for_seconds": (
                round(now - started, 1) if (_job["running"] and started) else None
            ),
            "result": _job["result"] if _job["result_at"] > since else None,
            "error": _job["error"] if _job["error_at"] > since else None,
        }


@app.get("/api/health")
def api_health() -> dict:
    """Small liveness probe used by the start script and the UI status dot."""
    return {
        "status": "ok",
        "model_loaded": _pipe_model,
        "models_available": len(list_models()),
        "free_ram_gb": round(_system_free_gb(), 2),
        "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "process_ram_gb": round(_process_ram_gb()[0], 2),
        "torch": torch.__version__,
    }


def main() -> None:  # pragma: no cover - convenience entry point
    import uvicorn

    ensure_dirs()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()

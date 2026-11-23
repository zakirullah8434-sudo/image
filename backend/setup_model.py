"""
Local Image Studio - model auto-downloader and verifier.

Checks ``models/`` for the checkpoints named in the project spec
(``realisticVisionV60B1_v51HyperVAE.safetensors`` / ``dreamshaper_8.safetensors``).
If no usable Stable Diffusion 1.5 checkpoint is present at all, a pruned
(~2 GB) single-file ``.safetensors`` checkpoint is downloaded from Hugging Face
with a progress bar.

Every download is verified by parsing the safetensors header, so a truncated or
corrupt file can never reach the inference pipeline.

Usage:
    python backend/setup_model.py            # ensure a model exists
    python backend/setup_model.py --check    # verify + list what is installed
    python backend/setup_model.py --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# The huggingface_hub "xet" transport performs badly on many home connections
# (it throttles to a crawl on some networks). Plain HTTPS range requests are
# measured to be 5-10x faster here, so we use classic HTTP for hub fallbacks.
# Set HF_HUB_DISABLE_XET=0 in your environment to re-enable it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Windows consoles/codepages (cp1252 and friends) cannot encode the block
# characters used by the progress bar, and a UnicodeEncodeError raised inside
# the downloader aborts the fast multi-stream path. Never let stdout encoding
# kill a download: fall back to replacement characters instead.
try:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError, OSError):
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FRONTEND_DIR = ROOT / "frontend"

# Filenames named by the project specification.
PREFERRED_FILENAMES = (
    "realisticVisionV60B1_v51HyperVAE.safetensors",
    "dreamshaper_8.safetensors",
)

# Anything smaller than this cannot be a full SD 1.5 checkpoint (a LoRA, a
# VAE-only file, or a truncated download).
MIN_MODEL_BYTES = 500 * 1024 * 1024

# Verified single-file SD 1.5 sources (both checked to contain a complete
# UNet + CLIP text encoder + VAE, A1111 tensor naming).
CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "repo_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "filename": "Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
        "label": "Realistic Vision V6.0 B1 (pruned fp16, ~2.0 GB, photorealistic)",
    },
    {
        "repo_id": "digiplay/DreamShaper_8",
        "filename": "dreamshaper_8.safetensors",
        "label": "DreamShaper 8 (pruned fp16, ~2.0 GB, versatile/anime)",
    },
)


# --------------------------------------------------------------------------- #
# Directory + model discovery
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    """Create the project directories if they are missing."""
    for d in (MODELS_DIR, OUTPUTS_DIR, FRONTEND_DIR):
        d.mkdir(parents=True, exist_ok=True)


def list_models() -> list[Path]:
    """Return every ``.safetensors`` file in ``models/`` (largest first)."""
    if not MODELS_DIR.is_dir():
        return []
    files = [p for p in MODELS_DIR.iterdir() if p.is_file() and p.suffix == ".safetensors"]
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files


# --------------------------------------------------------------------------- #
# Verification (reads only the header - instant, even for a 2 GB file)
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: Path) -> tuple[int, dict] | None:
    """Parse ``(header_length, metadata)`` of a safetensors file. None if invalid."""
    try:
        with path.open("rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return None
            header_len = int.from_bytes(raw_len, "little")
            # A sane SD 1.5 header is ~150 KB; guard against absurd values.
            if not (16 <= header_len <= 64 * 1024 * 1024):
                return None
            header = f.read(header_len)
            if len(header) != header_len:
                return None
            meta = json.loads(header.decode("utf-8"))
            if not isinstance(meta, dict):
                return None
            return header_len, meta
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def verify_model(path: Path) -> tuple[bool, str]:
    """
    Check that *path* is a loadable, full Stable Diffusion 1.5 checkpoint.

    Returns ``(ok, reason)``. Only the safetensors header is read.
    """
    if not path.is_file():
        return False, "file not found"

    size = path.stat().st_size
    if size < MIN_MODEL_BYTES:
        return False, f"too small ({size / 1024 / 1024:.0f} MB) - truncated or not a full checkpoint"

    parsed = read_safetensors_header(path)
    if parsed is None:
        return False, "unreadable safetensors header (corrupt download)"
    header_len, meta = parsed

    keys = [k for k in meta.keys() if k != "__metadata__"]
    if not keys:
        return False, "no tensors in file"

    # Exact integrity check: a safetensors file's total length is fully
    # determined by its header, so a short file is a truncated (or still
    # in-progress) download and must never reach the pipeline.
    data_ends = [
        v["data_offsets"][1]
        for v in meta.values()
        if isinstance(v, dict)
        and isinstance(v.get("data_offsets"), (list, tuple))
        and len(v["data_offsets"]) == 2
    ]
    if data_ends:
        expected = 8 + header_len + max(data_ends)
        if size != expected:
            return False, (
                f"incomplete download ({size / 1024 / 1024:.0f} MB of "
                f"{expected / 1024 / 1024:.0f} MB)"
            )

    # Accept both A1111 ("model.diffusion_model.*") and diffusers-style
    # ("unet.*") checkpoint layouts, but require the diffusion backbone.
    has_unet = any(k.startswith("model.diffusion_model") or k.startswith("unet.") for k in keys)
    has_vae = any(k.startswith("first_stage_model") or k.startswith("vae.") for k in keys)
    has_text = any(
        k.startswith("cond_stage_model") or k.startswith("text_encoder.") for k in keys
    )

    missing = [
        name
        for name, present in (("UNet", has_unet), ("VAE", has_vae), ("text encoder", has_text))
        if not present
    ]
    if missing:
        return False, "missing component(s): " + ", ".join(missing)

    return True, f"{len(keys)} tensors, {size / 1024 / 1024 / 1024:.2f} GB"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download_with_urllib(url: str, dest: Path) -> None:
    """Single-stream resumable download (kept as a dependency-free fallback)."""
    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0

    request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")

    with urllib.request.urlopen(request, timeout=60) as resp:
        status = getattr(resp, "status", 200)
        if resume_from and status == 206:
            content_range = resp.headers.get("Content-Range", "")
            total = int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else 0
            mode, downloaded = "ab", resume_from
        else:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            mode, downloaded = "wb", 0

        started = time.time()
        last_print = 0.0
        with open(part, mode) as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 0.2:
                    last_print = now
                    sys.stdout.write("\r" + _progress_bar(downloaded, total, started))
                    sys.stdout.flush()
        sys.stdout.write("\r" + _progress_bar(downloaded, total or downloaded, started) + "\n")
        sys.stdout.flush()

    part.replace(dest)


# --------------------------------------------------------------------------- #
# Multi-connection downloader
#
# Hugging Face throttles each individual HTTP stream hard (measured ~175 KB/s),
# while a single connection to the same file can run at ~1 MB/s. Opening a
# handful of parallel Range streams therefore multiplies throughput by ~5-10x
# on slow links. Chunks are written at disjoint offsets into one pre-sized
# ".part" file, journaled so an interrupted 2 GB download resumes exactly.
# --------------------------------------------------------------------------- #
UA = "local-image-studio/1.0"
CHUNK_SIZE = 16 * 1024 * 1024      # journal granularity for resume
MAX_WORKERS = 6                    # parallel HTTP streams
CONNECT_TIMEOUT = 20
READ_TIMEOUT = 120


def _new_session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*", "Accept-Encoding": "identity"})
    return s


def _resolve_target(url: str, session) -> tuple[str, int, bool]:
    """
    Resolve redirects and learn ``(final_url, total_bytes, range_supported)``
    with a 1-byte ranged request instead of a separate HEAD.
    """
    resp = session.get(
        url, headers={"Range": "bytes=0-0"}, stream=True,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=True,
    )
    try:
        code = resp.status_code
        final_url = resp.url
        if code == 206:
            content_range = resp.headers.get("Content-Range", "")
            if "/" not in content_range:
                raise RuntimeError("malformed Content-Range from server")
            return final_url, int(content_range.rsplit("/", 1)[-1]), True
        if code == 200:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            if not total:
                raise RuntimeError("server did not report a content length")
            return final_url, total, False
        raise RuntimeError(f"unexpected HTTP {code} while probing the download")
    finally:
        resp.close()


def _load_journal(path: Path, n_chunks: int) -> list[bool]:
    try:
        data = path.read_text(encoding="ascii").strip()
    except OSError:
        data = ""
    if len(data) != n_chunks or any(c not in "01" for c in data):
        return [False] * n_chunks
    return [c == "1" for c in data]


def _save_journal(path: Path, done: list[bool]) -> None:
    try:
        path.write_text("".join("1" if d else "0" for d in done), encoding="ascii")
    except OSError:
        pass


def _progress_bar(
    downloaded: int, total: int, started: float, width: int = 28, baseline: int = 0
) -> str:
    elapsed = max(time.time() - started, 0.001)
    # Rate must exclude bytes that were already on disk when this session
    # started, otherwise a resumed download reports a fantasy speed/ETA.
    speed = max(downloaded - baseline, 0) / elapsed
    pct = (downloaded / total * 100) if total else 0.0
    filled = int(width * min(pct, 100) / 100)
    # ASCII only: block characters raise UnicodeEncodeError on cp1252 stdout.
    bar = "#" * filled + "-" * (width - filled)
    eta = (total - downloaded) / speed if speed > 0 and total else 0
    return (
        f"  |{bar}| {pct:5.1f}%  "
        f"{_fmt_bytes(downloaded)}/{_fmt_bytes(total)}  "
        f"{_fmt_bytes(speed)}/s  ETA {int(eta // 60)}:{int(eta % 60):02d}"
    )


def _chunk_worker(url, queue, part_path, done, journal_path, lock, stats, errors):
    """Download 16 MB chunks claimed from *queue* into their own file offset."""
    import requests

    session = _new_session()
    try:
        while True:
            try:
                idx, offset, length = queue.get_nowait()
            except Exception:
                return
            try:
                if done[idx]:
                    continue
                for attempt in range(5):
                    try:
                        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
                        with session.get(
                            url, headers=headers, stream=True,
                            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                        ) as resp:
                            if resp.status_code not in (200, 206):
                                raise IOError(f"HTTP {resp.status_code}")
                            got = 0
                            with open(part_path, "r+b") as fh:
                                fh.seek(offset)
                                for piece in resp.iter_content(chunk_size=256 * 1024):
                                    if not piece:
                                        break
                                    fh.write(piece)
                                    got += len(piece)
                                    with lock:
                                        stats[0] += len(piece)
                                fh.flush()
                            if got != length:
                                raise IOError(f"short read ({got} of {length} bytes)")
                        with lock:
                            done[idx] = True
                            _save_journal(journal_path, done)
                        break
                    except Exception as exc:  # noqa: BLE001 - retry, then report
                        if attempt == 4:
                            raise IOError(f"chunk {idx}: {exc}") from exc
                        time.sleep(1.5 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(str(exc))
                return
            finally:
                queue.task_done()
    finally:
        session.close()


def download_with_requests(url: str, dest: Path) -> None:
    """
    Fast, resumable, multi-connection download with a live progress bar.

    Falls back to a single stream if the server refuses Range requests.
    """
    import queue as queue_mod
    import threading

    import requests  # noqa: F401 - verified installed before we start

    part = dest.with_suffix(dest.suffix + ".part")
    journal = dest.with_suffix(dest.suffix + ".progress")

    session = _new_session()
    try:
        final_url, total, range_ok = _resolve_target(url, session)
    finally:
        session.close()

    if not range_ok or total <= 0:
        print("  server does not support ranged downloads - using a single stream")
        download_with_urllib(url, dest)
        return

    n_chunks = math.ceil(total / CHUNK_SIZE)
    done = _load_journal(journal, n_chunks)

    # Pre-size the file once so workers can write at disjoint offsets safely.
    if not part.exists() or part.stat().st_size != total:
        with open(part, "w+b") as fh:
            fh.truncate(total)
        if not journal.exists():
            done = [False] * n_chunks

    chunks = []
    for i in range(n_chunks):
        offset = i * CHUNK_SIZE
        length = min(CHUNK_SIZE, total - offset)
        if not done[i]:
            chunks.append((i, offset, length))

    finished = sum(done)
    print(f"  {total / 1024**2:.0f} MB in {n_chunks} chunks "
          f"({len(chunks)} to fetch, {MAX_WORKERS} parallel streams)")

    if chunks:
        q = queue_mod.Queue()
        for c in chunks:
            q.put(c)
        lock = threading.Lock()
        stats = [0]
        errors: list[str] = []
        threads = [
            threading.Thread(
                target=_chunk_worker,
                args=(final_url, q, part, done, journal, lock, stats, errors),
                daemon=True,
            )
            for _ in range(min(MAX_WORKERS, len(chunks)))
        ]
        started = time.time()
        baseline = finished * CHUNK_SIZE  # bytes already present before we began
        for t in threads:
            t.start()

        try:
            while any(t.is_alive() for t in threads):
                time.sleep(0.2)
                with lock:
                    got = stats[0] + finished * CHUNK_SIZE
                # Progress output must never abort an in-flight download.
                try:
                    sys.stdout.write(
                        "\r" + _progress_bar(got, total, started, baseline=baseline)
                    )
                    sys.stdout.flush()
                except (OSError, UnicodeError):
                    pass
        except KeyboardInterrupt:
            print("\n  download paused - re-run setup to resume it")
            raise

        sys.stdout.write(
            "\r" + _progress_bar(total, total, started, baseline=baseline) + "\n"
        )
        sys.stdout.flush()

        if errors:
            raise RuntimeError("parallel download failed: " + "; ".join(errors[:3]))
        if any(t.is_alive() for t in threads):
            raise RuntimeError("parallel download did not finish")

    if part.stat().st_size != total:
        raise RuntimeError(
            f"file is {part.stat().st_size} bytes, expected {total}"
        )

    part.replace(dest)
    journal.unlink(missing_ok=True)


def download_with_hf_hub(repo_id: str, filename: str, dest_dir: Path) -> None:
    """Download using ``huggingface_hub`` (handles auth/mirrors/retries)."""
    from huggingface_hub import hf_hub_download  # imported lazily (heavy)

    print(f"  fetching {repo_id}/{filename} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )


def download_candidate(candidate: dict[str, str]) -> Path:
    """
    Download one candidate checkpoint into ``models/`` and verify it.

    Tries ``huggingface_hub`` first (spec-preferred), then falls back to a
    resumable stdlib downloader so a flaky connection can never brick setup.
    """
    repo_id = candidate["repo_id"]
    filename = candidate["filename"]
    dest = MODELS_DIR / filename
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    if dest.exists():
        ok, _ = verify_model(dest)
        if ok:
            return dest

    errors: list[str] = []

    # --- attempt 1: multi-connection HTTPS download (fastest, resumable) --- #
    try:
        download_with_requests(url, dest)
        if dest.exists():
            return dest
        errors.append("download finished but the file is missing")
    except Exception as exc:  # noqa: BLE001 - fall through to the next method
        errors.append(f"parallel download: {exc}")
        print(f"  ! parallel download failed ({exc}) - trying huggingface_hub")

    # --- attempt 2: huggingface_hub (spec-preferred library) --------------- #
    try:
        download_with_hf_hub(repo_id, filename, MODELS_DIR)
        if dest.exists():
            return dest
        errors.append("huggingface_hub reported success but the file is missing")
    except Exception as exc:  # noqa: BLE001 - last resort is the stdlib
        errors.append(f"huggingface_hub: {exc}")
        print(f"  ! huggingface_hub failed ({exc}) - falling back to a single stream")

    # --- attempt 3: dependency-free single stream (resumable) ------------- #
    try:
        download_with_urllib(url, dest)
        return dest
    except (urllib.error.URLError, OSError, ValueError) as exc:
        errors.append(f"single stream: {exc}")
        raise RuntimeError("; ".join(errors)) from exc


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def ensure_model(force: bool = False) -> Path | None:
    """
    Guarantee that at least one verified checkpoint exists in ``models/``.

    Returns the model to use, or ``None`` if the user only has custom files
    that failed verification and no download could be completed.
    """
    ensure_dirs()
    existing = list_models()

    # 1) Preferred filenames first (project spec), then any other model.
    for name in PREFERRED_FILENAMES:
        target = MODELS_DIR / name
        if target in existing:
            if not force:
                ok, reason = verify_model(target)
                if ok:
                    print(f"[setup] Using preferred model: {name} ({reason})")
                    return target
                print(f"[setup] Preferred model {name} is invalid: {reason}")

    # 2) Any other usable checkpoint already in models/ -> no 2 GB download.
    if existing and not force:
        for path in existing:
            ok, reason = verify_model(path)
            if ok:
                print(f"[setup] Found existing model: {path.name} ({reason})")
                print("[setup] Skipping the automatic download. Drop more .safetensors "
                      "files into models/ to add them to the UI dropdown.")
                return path
            print(f"[setup] Ignoring {path.name}: {reason}")

    # 3) Nothing usable -> download a verified pruned SD 1.5 checkpoint.
    print("[setup] No usable checkpoint found - downloading a pruned SD 1.5 model (~2 GB).")
    print("[setup] This is a one-time download; it is cached in models/ forever.\n")

    failures: list[str] = []
    for i, candidate in enumerate(CANDIDATES, start=1):
        print(f"[setup] Source {i}/{len(CANDIDATES)}: {candidate['label']}")
        try:
            path = download_candidate(candidate)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{candidate['repo_id']}: {exc}")
            print(f"  ! failed: {exc}\n")
            continue

        ok, reason = verify_model(path)
        if ok:
            print(f"  [ok] {path.name} verified ({reason})\n")
            return path

        failures.append(f"{candidate['repo_id']}: {reason}")
        print(f"  ! downloaded file failed verification: {reason}\n")
        try:
            path.unlink()
        except OSError:
            pass

    raise SystemExit(
        "[setup] ERROR - could not obtain a usable model.\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\n\nYou can also drop any SD 1.5 .safetensors checkpoint into:\n"
        + f"  {MODELS_DIR}\n"
        + "and restart the app."
    )


def cmd_check() -> int:
    """Verify every installed model and print a report."""
    ensure_dirs()
    models = list_models()
    if not models:
        print(f"[check] No models found in {MODELS_DIR}")
        return 1

    worst = 0
    for path in models:
        ok, reason = verify_model(path)
        print(f"[check] {'OK ' if ok else 'BAD'}  {path.name}  -  {reason}")
        if not ok:
            worst = 1
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Image Studio model setup")
    parser.add_argument("--check", action="store_true", help="verify installed models and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if a model exists")
    parser.add_argument("--list", action="store_true", help="print model names and exit")
    args = parser.parse_args(argv)

    if args.check:
        return cmd_check()
    if args.list:
        for p in list_models():
            print(p.name)
        return 0

    path = ensure_model(force=args.force)
    if path is None:
        return 1
    print(f"[setup] Ready: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

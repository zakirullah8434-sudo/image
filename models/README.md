# models/

Drop any **Stable Diffusion 1.5** `.safetensors` checkpoint in this folder and
it appears in the app's model dropdown automatically (click **↻ rescan**).

Nothing here is ever committed to git — checkpoints are ~2 GB, far above
GitHub's 100 MB per-file limit.

## Auto-download

```powershell
venv\Scripts\python.exe backend\setup_model.py
```

Downloads and verifies a pruned ~2 GB SD 1.5 checkpoint (~10 min on a slow
link). It is skipped whenever any valid `.safetensors` already exists here.

## Expected files

| File | Size |
|---|---|
| `Realistic_Vision_V6.0_NV_B1_fp16.safetensors` | 2,132,625,894 bytes |
| `dreamshaper_8.safetensors` | 2,132,625,894 bytes |
| anything else ending in `.safetensors` | scanned + listed |

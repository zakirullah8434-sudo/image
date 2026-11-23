#!/usr/bin/env bash
# ============================================================
#  Local Image Studio - one-click launcher (Linux / macOS)
#  Creates venv, installs CPU-only deps, downloads the model,
#  then serves the UI at http://127.0.0.1:8000
# ============================================================
set -e
cd "$(dirname "$0")"

VENV="venv"
PY="$VENV/bin/python"
HOST="127.0.0.1"
PORT="8000"

echo
echo " ============================================"
echo "   LOCAL IMAGE STUDIO  -  CPU AI Generator"
echo " ============================================"
echo

# [1/4] virtual environment
if [ ! -x "$PY" ]; then
    echo " [1/4] Creating virtual environment..."
    python3 -m venv "$VENV"
else
    echo " [1/4] Virtual environment found."
fi

# [2/4] dependencies
if [ ! -d "$VENV/lib/python3*/site-packages/torch" ] || [ ! -f "$VENV/.deps_ok" ]; then
    echo " [2/4] Installing CPU-only dependencies - first run only, about 2.5 GB."
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r backend/requirements.txt
    echo ok > "$VENV/.deps_ok"
else
    echo " [2/4] Dependencies ready."
fi

# [3/4] model
echo " [3/4] Checking for a Stable Diffusion checkpoint..."
"$PY" backend/setup_model.py

# [4/4] launch
echo
echo " [4/4] Starting the studio..."
echo "        URL : http://$HOST:$PORT"
echo "        Stop: Ctrl+C"
echo
"$PY" -m uvicorn main:app --host "$HOST" --port "$PORT" --app-dir backend

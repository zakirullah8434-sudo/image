@echo off
setlocal
title Local Image Studio
cd /d "%~dp0"

rem ============================================================
rem   Local Image Studio - one-click launcher (Windows)
rem   Creates venv, installs CPU-only deps, downloads the model,
rem   then serves the UI at http://127.0.0.1:8000
rem ============================================================

set "VENV=venv"
set "PY=%VENV%\Scripts\python.exe"
set "HOST=127.0.0.1"
set "PORT=8000"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo  ============================================
echo    LOCAL IMAGE STUDIO  -  CPU AI Generator
echo  ============================================
echo.

rem --- [1/4] virtual environment ------------------------------------
if not exist "%PY%" (
    echo  [1/4] Creating virtual environment...
    python -m venv "%VENV%"
    if not exist "%PY%" (
        echo.
        echo  ERROR: could not create the virtual environment.
        echo         Install Python 3.10+ and make sure "python" is on PATH.
        echo.
        pause
        exit /b 1
    )
) else (
    echo  [1/4] Virtual environment found.
)

rem --- [2/4] dependencies --------------------------------------------
if not exist "%VENV%\Lib\site-packages\torch" set "NEED_DEPS=1"
if not exist "%VENV%\.deps_ok" set "NEED_DEPS=1"

if defined NEED_DEPS (
    echo  [2/4] Installing CPU-only dependencies - first run only, about 2.5 GB.
    echo         torch / diffusers / transformers / fastapi ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r "backend\requirements.txt"
    if errorlevel 1 (
        echo.
        echo  ERROR: dependency installation failed. Check the messages above.
        echo.
        pause
        exit /b 1
    )
    echo ok> "%VENV%\.deps_ok"
) else (
    echo  [2/4] Dependencies ready.
)

rem --- [3/4] model ----------------------------------------------------
echo  [3/4] Checking for a Stable Diffusion checkpoint...
"%PY%" "backend\setup_model.py"
if errorlevel 1 (
    echo.
    echo  ERROR: model setup failed - see the messages above.
    echo         You can also drop any SD 1.5 .safetensors file into models\
    echo.
    pause
    exit /b 1
)

rem --- [4/4] launch ---------------------------------------------------
echo.
echo  [4/4] Starting the studio...
echo         URL : http://%HOST%:%PORT%
echo         Stop: close this window or press Ctrl+C
echo.

start "" "http://%HOST%:%PORT%"
"%PY%" -m uvicorn main:app --host %HOST% --port %PORT% --app-dir backend

echo.
echo  Server stopped.
pause

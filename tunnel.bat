@echo off
setlocal
title Local Image Studio - Public Tunnel
cd /d "%~dp0"

rem ============================================================
rem  Exposes the local backend (127.0.0.1:8000) on a public
rem  https:// address so the Vercel-hosted UI (or a phone on
rem  any network) can reach this PC.
rem
rem  Usage:  1) run start.bat     2) run this file
rem ============================================================

set "HOST=127.0.0.1"
set "PORT=8000"
set "BIN=tools\cloudflared.exe"
set "DL=https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

echo.
echo  ============================================
echo    LOCAL IMAGE STUDIO - PUBLIC TUNNEL
echo  ============================================
echo.

rem --- is the backend up? ------------------------------------------------
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo  [!] Nothing is listening on port %PORT%.
    echo      Run start.bat FIRST, keep it open, then run this file.
    echo.
    pause
    exit /b 1
)

rem --- fetch cloudflared once -------------------------------------------
if exist "%BIN%" (
    echo  [1/2] cloudflared already downloaded.
) else (
    echo  [1/2] Downloading cloudflared - one time, about 40 MB...
    powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path 'tools' ^| Out-Null; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%DL%' -OutFile '%BIN%' -UseBasicParsing"
    if not exist "%BIN%" (
        echo.
        echo  ERROR: download failed. Get it manually from
        echo         https://github.com/cloudflare/cloudflared/releases
        echo         and save it as:  tools\cloudflared.exe
        echo.
        pause
        exit /b 1
    )
)

echo  [2/2] Starting tunnel to http://%HOST%:%PORT% ...
echo.
echo  ---------------------------------------------------------------
echo   1. Copy the  https://....trycloudflare.com  address below.
echo   2. Open your hosted app (Vercel) or the local page.
echo   3. Click the  "local"  button in the top bar, paste the
echo      address, press Connect.
echo.
echo   Keep THIS window open - closing it stops the tunnel.
echo  ---------------------------------------------------------------
echo.

"%BIN%" tunnel --url http://%HOST%:%PORT% --no-autoupdate

echo.
echo  Tunnel closed.
pause

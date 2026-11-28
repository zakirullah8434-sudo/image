@echo off
setlocal
title Local Image Studio (auto)
cd /d "%~dp0"

rem ============================================================
rem  Local Image Studio - starts automatically with Windows.
rem
rem  1) launches the backend if it is not already up
rem  2) waits until it answers /api/health (torch import ~60 s)
rem  3) starts the public tunnel if cloudflared is not running
rem  4) loops forever and restarts anything that died
rem
rem  Installed into shell:startup by the setup (see README.md).
rem  Stop it with stop.bat - it comes back at the next sign-in.
rem ============================================================

if not exist "venv\Scripts\python.exe" (
    echo  [auto] venv not found - run start.bat once to finish setup.
    exit /b 1
)

echo [%date% %time%] autorun started>>"%TEMP%\lis-autorun.log"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:LOOP
rem ---- backend -------------------------------------------------
netstat -ano | findstr "LISTENING" | findstr ":8000" >nul
if errorlevel 1 (
    echo  [auto] starting backend...
    echo [%date% %time%] starting backend>>"%TEMP%\lis-autorun.log"
    start "" /b "venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend
)

rem ---- wait for health (first start can take ~60 s) -------------
set /a _wait=0
:HEALTH
curl.exe -s -m 3 http://127.0.0.1:8000/api/health >nul 2>&1
if not errorlevel 1 goto HEALTH_OK
set /a _wait+=1
if %_wait% lss 60 (
    ping -n 4 127.0.0.1 >nul
    goto HEALTH
)
echo  [auto] backend not ready yet - will retry in 30 s.
goto SLEEP

:HEALTH_OK
rem ---- tunnel ---------------------------------------------------
tasklist /fi "imagename eq cloudflared.exe" 2>nul | findstr /i "cloudflared" >nul
if errorlevel 1 (
    echo  [auto] starting tunnel...
    echo [%date% %time%] starting tunnel>>"%TEMP%\lis-autorun.log"
    set "LIS_AUTORUN=1"
    start "" /b powershell -NoProfile -ExecutionPolicy Bypass -File "tunnel.ps1"
)

:SLEEP
rem ---- re-check everything every 30 s ---------------------------
ping -n 31 127.0.0.1 >nul
goto LOOP

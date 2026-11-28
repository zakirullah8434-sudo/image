@echo off
setlocal
title Local Image Studio - auto-start setup
cd /d "%~dp0"

rem ============================================================
rem  ONE-TIME SETUP - double-click this once.
rem
rem  Makes the backend + public tunnel start with Windows and
rem  keep themselves running (30 s watchdog). After this you
rem  never need to open start.bat or tunnel.bat again - just
rem  turn the PC on and the hosted page connects by itself.
rem
rem  Stop everything : stop.bat
rem  Logs            : %TEMP%\lis-autorun.log
rem ============================================================

echo.
echo  ============================================
echo    LOCAL IMAGE STUDIO - AUTO-START SETUP
echo  ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-autostart.ps1"
if errorlevel 1 (
    echo.
    echo  ERROR: could not create the Startup shortcut.
    echo         Make sure you run this as a normal logged-in user.
    echo.
    pause
    exit /b 1
)

rem Start the watchdog right now (hidden) - works without signing out.
wscript "%~dp0autorun.vbs"

echo.
echo  Done! Local Image Studio now starts automatically with Windows:
echo.
echo    - backend on  http://127.0.0.1:8000
echo    - public tunnel for the hosted page and your phone
echo    - restarts either one within ~30 s if it ever dies
echo.
echo  First start after boot takes about 1 minute (model server warms up).
echo.
echo    Logs : %TEMP%\lis-autorun.log
echo    Stop : stop.bat   (auto-start returns at the next sign-in)
echo.
pause

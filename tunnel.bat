@echo off
setlocal
title Local Image Studio - Public Tunnel
cd /d "%~dp0"

rem ============================================================
rem  Exposes the local backend (127.0.0.1:8000) on a public
rem  https:// address so the Vercel-hosted UI (or a phone on
rem  any network) can reach this PC.  The address is published
rem  to the website automatically (see tunnel.ps1) - the hosted
rem  page connects to it by itself, nothing to paste.
rem
rem  Usage:  1) run start.bat     2) run this file
rem ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tunnel.ps1"

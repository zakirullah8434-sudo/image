# ============================================================
#  Local Image Studio - stop everything (backend + tunnel +
#  the autorun watchdog).  Run via stop.bat.
# ============================================================
$ErrorActionPreference = "SilentlyContinue"

# 1) the autorun watchdog first, so it cannot restart anything
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
    Where-Object { $_.CommandLine -like "*autorun.bat*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# 2) tunnel (powershell running tunnel.ps1) + cloudflared
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*tunnel.ps1*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force

# 3) the backend on port 8000
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

Write-Host ""
Write-Host " Local Image Studio stopped (backend + tunnel + auto-start)."
Write-Host " It starts again at the next Windows sign-in, or run start.bat."
Write-Host ""

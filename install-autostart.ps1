# ============================================================
#  Local Image Studio - one-time auto-start setup.
#  Creates a shortcut in the Windows Startup folder that runs
#  autorun.vbs (hidden) at every sign-in -> backend + tunnel
#  come up automatically and stay up (watchdog restarts them).
# ============================================================
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $root) { $root = (Get-Location).Path }

$startup = [Environment]::GetFolderPath("Startup")
if (-not (Test-Path $startup)) {
    Write-Host " [!] Startup folder not found: $startup"
    exit 1
}

$vbs = Join-Path $root "autorun.vbs"
if (-not (Test-Path $vbs)) {
    Write-Host " [!] autorun.vbs is missing next to this script."
    exit 1
}

$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $startup "Local Image Studio.lnk"))
$lnk.TargetPath = "wscript.exe"
$lnk.Arguments = '"' + $vbs + '"'
$lnk.WorkingDirectory = $root
$lnk.WindowStyle = 7
$lnk.Description = "Local Image Studio - backend + tunnel auto-start"
$lnk.Save()

Write-Host " [+] Startup shortcut created:"
Write-Host "     $startup\Local Image Studio.lnk"
exit 0

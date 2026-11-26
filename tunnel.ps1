# ============================================================
#  Local Image Studio - public tunnel (cloudflared quick tunnel)
#
#  Starts a free https://....trycloudflare.com tunnel to the local
#  backend and PUBLISHES that address to the app automatically:
#  it writes frontend/backend.json, commits and pushes, and the
#  hosted page (Vercel) picks the address up and connects itself.
#  Nothing to copy, nothing to paste.
#
#  Usage: run tunnel.bat and keep the window open.
# ============================================================

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $root) { $root = (Get-Location).Path }
Set-Location $root

$HostAddr = "127.0.0.1"
$Port     = 8000
$Bin      = Join-Path $root "tools\cloudflared.exe"
$Dl       = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

Write-Host ""
Write-Host " ============================================"
Write-Host "   LOCAL IMAGE STUDIO - PUBLIC TUNNEL"
Write-Host " ============================================"
Write-Host ""

# --- is the backend up? -----------------------------------------------
$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Write-Host " [!] Nothing is listening on port $Port."
    Write-Host "     Run start.bat FIRST, keep it open, then run this file."
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}

# --- fetch cloudflared once --------------------------------------------
if (Test-Path $Bin) {
    Write-Host " [1/2] cloudflared already downloaded."
} else {
    Write-Host " [1/2] Downloading cloudflared - one time, about 40 MB..."
    try {
        New-Item -ItemType Directory -Force -Path (Join-Path $root "tools") | Out-Null
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Dl -OutFile $Bin -UseBasicParsing
    } catch { }
    if (-not (Test-Path $Bin)) {
        Write-Host ""
        Write-Host "  ERROR: download failed. Get it manually from"
        Write-Host "         https://github.com/cloudflare/cloudflared/releases"
        Write-Host "         and save it as:  tools\cloudflared.exe"
        Write-Host ""
        Read-Host "Press Enter to close"
        exit 1
    }
}

Write-Host " [2/2] Starting tunnel to http://${HostAddr}:$Port ..."
Write-Host ""
Write-Host " ---------------------------------------------------------------"
Write-Host "  The https://....trycloudflare.com address is published to the"
Write-Host "  website AUTOMATICALLY - the hosted page connects by itself in"
Write-Host "  about a minute. Nothing to copy or paste."
Write-Host ""
Write-Host "  Keep THIS window open - closing it stops the tunnel."
Write-Host " ---------------------------------------------------------------"
Write-Host ""

# Write the address where both copies of the site can read it, then push.
function Publish-TunnelUrl([string]$Url) {
    $jsonPath = Join-Path $root "frontend\backend.json"
    $payload  = '{"url":"' + $Url + '","updated":"' + (Get-Date -Format o) + '"}'
    try {
        # UTF8 without BOM so res.json() always parses it.
        [System.IO.File]::WriteAllText($jsonPath, $payload, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host " [+] Tunnel address: $Url"
    } catch {
        Write-Host " [!] Could not write frontend\backend.json - $($_.Exception.Message)"
        return
    }
    # Commit ONLY this file, then push: GitHub -> Vercel rebuild (~1 min).
    # The page also reads the raw GitHub copy, which updates in seconds.
    # Self-heal: a run that was killed mid-`git add` can leave .git/index
    # corrupt ("bad signature 0x00000000"), breaking every later command.
    # Detect that here and rebuild the index before staging anything.
    git status --porcelain 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host " [!] Git index was corrupt - rebuilding it..." -ForegroundColor Yellow
        Remove-Item (Join-Path $root ".git\index") -Force -ErrorAction SilentlyContinue
        git status --porcelain 2>&1 | Out-Null
    }
    git add frontend/backend.json 2>&1 | Out-Null
    git commit -m "chore: refresh tunnel address" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        git push origin main 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [+] Published - the hosted page will connect automatically."
        } else {
            Write-Host " [!] git push failed - the hosted page cannot see the new" -ForegroundColor Yellow
            Write-Host "     address. Paste it manually in the app instead." -ForegroundColor Yellow
        }
    } else {
        Write-Host " [!] git commit failed - the hosted page cannot see the" -ForegroundColor Yellow
        Write-Host "     address. Paste it manually in the app instead." -ForegroundColor Yellow
    }
}

# Run cloudflared, echoing every line, and publish the first URL we see.
$script:published = $false
& $Bin tunnel --url "http://${HostAddr}:$Port" --no-autoupdate 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    if (-not $script:published -and $line -match 'https://[a-z0-9\-]+\.trycloudflare\.com') {
        $script:published = $true
        Publish-TunnelUrl $matches[0]
    }
}

Write-Host ""
Write-Host " Tunnel closed."
Read-Host "Press Enter to close"

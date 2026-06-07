# VCAI demo launcher
#
# What it does:
#   1. Optionally rebuilds the frontend (skip with -SkipBuild)
#   2. Starts the FastAPI backend on port 8000 (in a new window)
#   3. Waits for /health to be reachable
#   4. Starts a Cloudflare quick tunnel pointing at port 8000
#   5. Prints the public https://*.trycloudflare.com URL you can share
#
# Usage (run from the project root):
#   .\scripts\start_demo.ps1                 # rebuild + serve + tunnel
#   .\scripts\start_demo.ps1 -SkipBuild      # skip the npm rebuild (faster)
#   .\scripts\start_demo.ps1 -BackendOnly    # no tunnel (local-only test)
#
# Stop the tunnel by pressing Ctrl+C in this window. The backend keeps
# running in its own window.

param(
    [switch]$SkipBuild,
    [switch]$BackendOnly
)

$ErrorActionPreference = 'Continue'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host " VCAI demo launcher" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Frontend build (optional) ------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "[1/3] Building frontend..." -ForegroundColor Yellow
    Push-Location frontend
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Frontend build failed. Aborting." -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location
    Write-Host "Frontend built." -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[1/3] Skipping frontend build (-SkipBuild)." -ForegroundColor DarkGray
    Write-Host ""
}

# 2. Backend ------------------------------------------------------------------
Write-Host "[2/3] Starting backend on http://localhost:8000 ..." -ForegroundColor Yellow

$pythonExe = "C:\Users\11\anaconda3\envs\vcai\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$backendArgs = "-m", "backend.main"
$backendProc = Start-Process -FilePath $pythonExe `
                             -ArgumentList $backendArgs `
                             -WorkingDirectory $ProjectRoot `
                             -PassThru `
                             -WindowStyle Normal
Write-Host ("Backend started (PID=" + $backendProc.Id + "). Wait ~30s for models to load.") -ForegroundColor Green
Write-Host ""

Write-Host "Waiting for backend to be ready..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        # not ready yet
    }
    Start-Sleep -Seconds 1
    if ($i -eq 30) { Write-Host "  ...still loading (this can take ~40s with models warming)" -ForegroundColor DarkGray }
}

if (-not $ready) {
    Write-Host "Backend did not become healthy within 90s. Check the backend window for errors." -ForegroundColor Red
    Write-Host "You can still continue. Start the tunnel manually with:" -ForegroundColor Yellow
    Write-Host "  .\tools\cloudflared.exe tunnel --url http://localhost:8000" -ForegroundColor Cyan
    exit 1
}
Write-Host "Backend ready." -ForegroundColor Green
Write-Host ""

if ($BackendOnly) {
    Write-Host "Backend-only mode. Visit http://localhost:8000" -ForegroundColor Cyan
    Write-Host "(Backend stays running in its window. Close that window to stop.)" -ForegroundColor DarkGray
    exit 0
}

# 3. Cloudflare quick tunnel --------------------------------------------------
Write-Host "[3/3] Starting Cloudflare tunnel ..." -ForegroundColor Yellow
$cloudflared = Join-Path $ProjectRoot "tools\cloudflared.exe"
if (-not (Test-Path $cloudflared)) {
    Write-Host ("cloudflared.exe not found at " + $cloudflared) -ForegroundColor Red
    Write-Host "Download from https://github.com/cloudflare/cloudflared/releases" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "When you see a line like" -ForegroundColor DarkGray
Write-Host "    https://random-words.trycloudflare.com" -ForegroundColor Cyan
Write-Host "that is your public demo URL. Share that with your friend." -ForegroundColor DarkGray
Write-Host ""
Write-Host "Press Ctrl+C to stop the tunnel (the backend keeps running)." -ForegroundColor DarkGray
Write-Host "-------------------------------------------------------------------" -ForegroundColor Cyan

# --protocol http2: TCP-based transport. The default (QUIC/UDP) drops idle
# connections on flaky WiFi ("timeout: no recent network activity"). HTTP/2
# is far more stable for long-running demo tunnels.
& $cloudflared tunnel --url http://localhost:8000 --protocol http2

Write-Host ""
Write-Host "Tunnel stopped." -ForegroundColor Yellow
Write-Host ("Backend is still running in its window. Close it or kill PID " + $backendProc.Id + " to stop.") -ForegroundColor DarkGray

# PowerShell script to run smoke test with API server

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "DPDP COMPLIANCE API - SMOKE TEST RUNNER" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

# Check if API is already running
Write-Host "[INFO] Checking if API is already running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "  ✓ API is already running" -ForegroundColor Green
    $apiAlreadyRunning = $true
} catch {
    Write-Host "  ✗ API not running, will start it" -ForegroundColor Yellow
    $apiAlreadyRunning = $false
}

$apiProcess = $null

if (-not $apiAlreadyRunning) {
    Write-Host ""
    Write-Host "[INFO] Starting API server..." -ForegroundColor Yellow
    Write-Host "  Command: python src/api/server.py" -ForegroundColor Gray
    
    # Start API in background
    $apiProcess = Start-Process -FilePath "python" -ArgumentList "src/api/server.py" -PassThru -WindowStyle Hidden
    
    Write-Host "  Waiting for API to initialize (30s)..." -ForegroundColor Gray
    Start-Sleep -Seconds 30
    
    # Check if API started successfully
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Host "  ✓ API started successfully (PID: $($apiProcess.Id))" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ API failed to start" -ForegroundColor Red
        if ($apiProcess -and -not $apiProcess.HasExited) {
            Stop-Process -Id $apiProcess.Id -Force
        }
        exit 1
    }
}

# Run smoke test
Write-Host ""
Write-Host "[INFO] Running smoke test..." -ForegroundColor Yellow
Write-Host ""

python test_smoke_parity.py
$testExitCode = $LASTEXITCODE

# Cleanup: stop API if we started it
if (-not $apiAlreadyRunning -and $apiProcess) {
    Write-Host ""
    Write-Host "[INFO] Stopping API server (PID: $($apiProcess.Id))..." -ForegroundColor Yellow
    try {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  ✓ API server stopped" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Could not stop API server" -ForegroundColor Yellow
    }
}

Write-Host ""
exit $testExitCode

# Run from repo-root PowerShell: .\scripts\wsl\stop_demo.ps1
param([ValidateRange(1024, 65535)][int]$Port = 8765)
$ErrorActionPreference = 'Stop'
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/shutdown" -Method Post -TimeoutSec 3 | Out-Null
    Write-Output 'DeltaShot local demo stopped.'
} catch {
    Write-Output "No DeltaShot demo server responded on port $Port."
}

# Run from repo-root PowerShell: .\scripts\wsl\run_demo.ps1
param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [ValidateRange(10, 1800)][int]$StartupTimeoutSeconds = 600
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoForWsl = $repo.Replace([char]92, [char]47)
$wslPathOutput = & wsl.exe -d Ubuntu -- wslpath -u $repoForWsl 2>&1
$wslPathExitCode = $LASTEXITCODE
if ($wslPathExitCode -ne 0) {
    $details = ($wslPathOutput | Out-String).Trim()
    throw "Could not convert the repository path into WSL (exit $wslPathExitCode). $details"
}
$wslRepo = ($wslPathOutput | Out-String).Trim()
if ([string]::IsNullOrWhiteSpace($wslRepo)) {
    throw 'wslpath returned an empty repository path.'
}
$url = "http://127.0.0.1:$Port/"
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
    if ($health.ready -and $health.api_version -eq 3 -and $health.neural_available) {
        Start-Process $url
        Write-Output "DeltaShot demo already running: $url"
        exit 0
    }
    if ($health.ready) {
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/shutdown" -Method Post -TimeoutSec 3 | Out-Null
        Start-Sleep -Seconds 2
    }
} catch { }

$results = Join-Path $repo 'results'
New-Item -ItemType Directory -Force -Path $results | Out-Null
$stdout = Join-Path $results 'local_demo_server.log'
$stderr = Join-Path $results 'local_demo_server.err.log'
$wslScript = "$wslRepo/scripts/wsl/serve_demo.sh"
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList @(
    '-d', 'Ubuntu', '--', 'bash', $wslScript, [string]$Port
) -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
$ready = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($health.ready -and $health.api_version -eq 3 -and $health.neural_available) { $ready = $true; break }
    } catch {
        if ($process.HasExited) { break }
    }
}
if (-not $ready) {
    $process.Refresh()
    Write-Output "DeltaShot demo did not become ready within $StartupTimeoutSeconds seconds (wsl.exe exited=$($process.HasExited)). Logs: $stdout and $stderr"
    if (Test-Path $stdout) { Get-Content $stdout -Tail 30 }
    if (Test-Path $stderr) { Get-Content $stderr -Tail 30 }
    exit 1
}
Set-Content -Path (Join-Path $results 'local_demo_server.pid') -Value $process.Id
Start-Process $url
Write-Output "DeltaShot neural demo ready at $url (CPU, epoch 27). Server logs: $stdout"

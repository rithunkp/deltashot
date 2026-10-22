# Run from repo-root PowerShell: .\scripts\wsl\verify_demo.ps1
param(
    [ValidateRange(1, 65535)][int]$Port = 8765,
    [ValidateRange(30, 1800)][int]$StartupTimeoutSeconds = 600
)
$ErrorActionPreference = 'Stop'
$repo = 'C:\Rithun\Github\deltashot'
$premadeManifest = Join-Path $repo 'results\premade_demos\manifest.json'
if (-not (Test-Path -LiteralPath $premadeManifest -PathType Leaf)) {
    throw 'Premade results are not ready. First run .\scripts\wsl\build_premade_demos.ps1 and wait until results\premade_demos\manifest.json exists.'
}
$job = 'demo_verify'
$stdoutPath = Join-Path $repo 'results\demo_verify_wsl_stdout.log'
$stderrPath = Join-Path $repo 'results\demo_verify_wsl_stderr.log'
New-Item -ItemType Directory -Path (Join-Path $repo 'results') -Force | Out-Null
$wslArgs = @('-d', 'Ubuntu', '--', 'bash', '/mnt/c/Rithun/Github/deltashot/scripts/wsl/run_logged.sh',
    $job, '.venv_wsl/bin/python', '-u', '-m', 'scripts.verify_demo',
    '--port', [string]$Port, '--startup-timeout', [string]$StartupTimeoutSeconds)
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArgs -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Write-Output "Started detached demo verification in WSL (PID $($process.Id)). Follow results\demo_verify.log. When it finishes, paste results\demo_verify_report.txt and results\demo_verify_report.json. Browser screenshots: results\demo_verify_live.png and results\demo_verify_premade.png."

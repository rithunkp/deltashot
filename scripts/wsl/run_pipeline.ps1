# Run from the repo root in PowerShell: .\scripts\wsl\run_pipeline.ps1
param(
    [ValidateRange(0.1, 72)][double]$BudgetHours = 8,
    [string[]]$Stages = @(),
    [switch]$CalibrateOnly,
    [switch]$Resume,
    [switch]$SkipPytest
)
$ErrorActionPreference = 'Stop'
$repo = 'C:\Rithun\Github\deltashot'
$results = Join-Path $repo 'results'
New-Item -ItemType Directory -Path $results -Force | Out-Null
$statusPath = Join-Path $results 'pipeline_status.txt'
if (-not $Resume) {
    [System.IO.File]::WriteAllText($statusPath, '')
} elseif (-not (Test-Path -LiteralPath $statusPath)) {
    [System.IO.File]::WriteAllText($statusPath, '')
}
$job = 'pipeline'
$wslArgs = @('-d', 'Ubuntu', '--', 'bash', '/mnt/c/Rithun/Github/deltashot/scripts/wsl/run_logged.sh',
    $job, '.venv_wsl/bin/python', '-u', '-m', 'scripts.run_pipeline',
    '--device', 'auto', '--budget-hours', [string]$BudgetHours)
if ($Stages.Count -gt 0) { $wslArgs += '--stages'; $wslArgs += $Stages }
if ($CalibrateOnly) { $wslArgs += '--calibrate-only' }
if ($Resume) { $wslArgs += '--resume' }
if ($SkipPytest) { $wslArgs += '--skip-pytest' }
$stdoutPath = Join-Path $results 'pipeline_wsl_stdout.log'
$stderrPath = Join-Path $results 'pipeline_wsl_stderr.log'
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArgs -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Write-Output "Started detached wsl.exe (PID $($process.Id); $BudgetHours h budget; resume=$Resume). Watch results/pipeline.log, results/pipeline_pytest.txt, and results/pipeline_status.txt. WSL startup errors: results/pipeline_wsl_stderr.log."

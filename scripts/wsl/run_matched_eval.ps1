# Run from repo-root PowerShell: .\scripts\wsl\run_matched_eval.ps1 -Epoch 9 -PerBucket 50
# This competes for CPU with training; keep the stratified subset small.
param(
    [Parameter(Mandatory = $true)][ValidateRange(0, 27)][int]$Epoch,
    [ValidateRange(1, 400)][int]$PerBucket = 50,
    [string]$NoRefCheckpoint = '',
    [string]$RefCheckpoint = '',
    [string]$LongerNoRefCheckpoint = ''
)
$ErrorActionPreference = 'Stop'
$repo = 'C:\Rithun\Github\deltashot'
$job = "matched_eval_epoch_$Epoch"
$wslArgs = @('-d', 'Ubuntu', '--', 'bash', '/mnt/c/Rithun/Github/deltashot/scripts/wsl/run_logged.sh',
    $job, '.venv_wsl/bin/python', '-u', '-m', 'scripts.run_pipeline',
    '--device', 'auto', '--stages', 'evaluate', 'mode_selection', 'report',
    '--resume', '--epoch', [string]$Epoch, '--eval-pairs-per-bucket', [string]$PerBucket)
if ($NoRefCheckpoint) { $wslArgs += @('--noref-checkpoint', $NoRefCheckpoint) }
if ($RefCheckpoint) { $wslArgs += @('--ref-checkpoint', $RefCheckpoint) }
if ($LongerNoRefCheckpoint) { $wslArgs += @('--longer-noref-checkpoint', $LongerNoRefCheckpoint) }
$stdoutPath = Join-Path $repo "results\${job}_wsl_stdout.log"
$stderrPath = Join-Path $repo "results\${job}_wsl_stderr.log"
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArgs -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Write-Output "Started detached WSL matched evaluation (PID $($process.Id)); epoch=$Epoch, epochs completed=$($Epoch + 1), $PerBucket pairs/bucket. This competes with CPU training. Watch results\${job}.log and results\matched_epoch_${Epoch}_summary.json."

# Request a safe pause at the end of the current training epoch.
$ErrorActionPreference = 'Stop'
$repo = 'C:\Rithun\Github\deltashot'
$results = Join-Path $repo 'results'
if (-not (Test-Path -LiteralPath $results)) {
    throw "Results directory does not exist: $results"
}
$marker = Join-Path $results 'STOP_AFTER_EPOCH'
New-Item -ItemType File -Path $marker -Force | Out-Null
Write-Output 'Pause requested. The active trainer will stop after its current epoch checkpoint is saved.'

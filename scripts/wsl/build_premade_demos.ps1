# Run from repo-root PowerShell: .\scripts\wsl\build_premade_demos.ps1
param()
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$results = Join-Path $repo 'results'
New-Item -ItemType Directory -Path $results -Force | Out-Null
$stdout = Join-Path $results 'premade_demos_wsl_stdout.log'
$stderr = Join-Path $results 'premade_demos_wsl_stderr.log'
$jobLog = Join-Path $results 'build_premade_demos.log'
Set-Content -Path $jobLog -Value '' -NoNewline
$wslArgs = @('-d', 'Ubuntu', '--', 'bash', '/mnt/c/Rithun/Github/deltashot/scripts/wsl/run_logged.sh',
    'build_premade_demos', '.venv_wsl/bin/python', '-u', '-m', 'scripts.build_premade_demos')
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArgs -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output "Started premade demo build in WSL (PID $($process.Id)). Watch results\build_premade_demos.log. Wait for 'wrote 18 premade comparisons' and confirm results\premade_demos\manifest.json exists before verification. WSL stderr: $stderr"

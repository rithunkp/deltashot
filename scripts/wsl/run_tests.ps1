# Run from the repo root in PowerShell: .\scripts\wsl\run_tests.ps1
$ErrorActionPreference = 'Stop'
$line = 'start "" /b wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Rithun/Github/deltashot && .venv_wsl/bin/python -m pytest -q > results/test_output.txt 2>&1"'
Start-Process -FilePath $env:ComSpec -ArgumentList @('/c', $line) -WindowStyle Hidden
Write-Output 'Started detached WSL pytest. Read results/test_output.txt for the result.'

# Run from the repo root in PowerShell: .\scripts\wsl\run_throughput_probe.ps1
$ErrorActionPreference = 'Stop'
$repoWsl = '/mnt/c/Rithun/Github/deltashot'
$line = 'start "" /b wsl.exe -d Ubuntu -- bash {0}/scripts/wsl/run_logged.sh throughput_probe .venv_wsl/bin/python -u -m scripts.train_baseline --dataset data/synthetic_pairs_full --epochs 1 --batch-size 4 --patch-size 256 --lmbda 0.0130 --device cpu --save-dir checkpoints/throughput_probe --max-pairs-per-bucket 300 --max-steps 300 --log-every 1 --metrics-output results/throughput_probe_metrics.json' -f $repoWsl
Start-Process -FilePath $env:ComSpec -ArgumentList @('/c', $line) -WindowStyle Hidden
Write-Output 'Started detached throughput probe. Follow results/throughput_probe.log and results/throughput_probe_metrics.json.'

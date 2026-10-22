# Run from the repo root in PowerShell: .\scripts\wsl\run_phase3_train.ps1 -Lambda 0.013 -Epochs 30 -PairsPerEpoch 20000 [-Resume]
param(
    [Parameter(Mandatory = $true)][ValidateRange(0.00001, 10)][double]$Lambda,
    [Parameter(Mandatory = $true)][ValidateRange(1, 10000)][int]$Epochs,
    [Parameter(Mandatory = $true)][ValidateRange(5, 20000)][int]$PairsPerEpoch,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
if (($PairsPerEpoch % 5) -ne 0) { throw 'PairsPerEpoch must be divisible by five to preserve equal bucket balance.' }
$pairsPerBucket = [int]($PairsPerEpoch / 5)
$tag = $Lambda.ToString('0.00000', [Globalization.CultureInfo]::InvariantCulture).Replace('.', '')
$job = "phase3_lambda_$tag"
$trainArgs = @(
    '.venv_wsl/bin/python -u -m scripts.train_baseline'
    '--dataset data/synthetic_pairs_full'
    "--epochs $Epochs"
    '--batch-size 4'
    '--patch-size 256'
    "--lmbda $($Lambda.ToString('0.00000', [Globalization.CultureInfo]::InvariantCulture))"
    '--device cpu'
    "--save-dir checkpoints/$job"
    "--max-pairs-per-bucket $pairsPerBucket"
    "--metrics-output results/${job}_metrics.json"
)
if ($Resume) { $trainArgs += '--resume' }
$line = 'start "" /b wsl.exe -d Ubuntu -- bash /mnt/c/Rithun/Github/deltashot/scripts/wsl/run_logged.sh {0} {1}' -f $job, ($trainArgs -join ' ')
Start-Process -FilePath $env:ComSpec -ArgumentList @('/c', $line) -WindowStyle Hidden
Write-Output "Started detached $job. Check results/$job.log and checkpoints/$job/."

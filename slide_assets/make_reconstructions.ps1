param(
    [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoForWsl = $repo.Replace('\', '/')
$wslRepo = (& wsl.exe -d $Distro -- wslpath -u $repoForWsl 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($wslRepo -join ""))) {
    throw "Could not convert repo path into WSL. Confirm distro '$Distro' is available."
}
$wslRepo = ($wslRepo -join "").Trim()
$shellCommand = "cd '$wslRepo' && .venv_wsl/bin/python -u slide_assets/make_reconstructions.py > slide_assets/make_reconstructions.log 2>&1"
$arguments = "-d $Distro -- bash -lc `"$shellCommand`""
$process = Start-Process -FilePath "wsl.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru
Write-Host "Started detached WSL reconstruction montage (PID $($process.Id)). Watch slide_assets/make_reconstructions.log; the script uses existing evaluation decodes and does not train or load checkpoints."

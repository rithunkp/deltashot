# Run from the repo root: .\scripts\hf\launch_job.ps1 -BundleRepo USERNAME/deltashot-bundle -ArtifactRepo USERNAME/deltashot-results -Mode dryrun -Flavor cpu-upgrade
param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[^/]+/[^/]+$')][string]$BundleRepo,
    [Parameter(Mandatory = $true)][ValidatePattern('^[^/]+/[^/]+$')][string]$ArtifactRepo,
    [ValidateSet('dryrun', 'calibrate', 'full')][string]$Mode = 'calibrate',
    [ValidateSet('t4-small', 'cpu-upgrade')][string]$Flavor = 't4-small'
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python is not on PATH. Install huggingface_hub into the Python environment used to submit jobs.'
}
if (-not $env:HF_TOKEN) { throw 'Set HF_TOKEN in the environment. It is sent as a Job secret and never printed.' }

$arguments = @(
    'scripts/hf/launch_job.py',
    '--bundle-repo', $BundleRepo,
    '--artifact-repo', $ArtifactRepo,
    '--mode', $Mode,
    '--flavor', $Flavor
)
& python @arguments
if ($LASTEXITCODE -ne 0) { throw "Hugging Face Job submission failed with exit code $LASTEXITCODE" }

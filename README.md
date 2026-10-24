# DeltaShot

DeltaShot is a research prototype for compressing UI screenshots when a related reference screenshot may already be available. It compares conventional image codecs with learned no-reference and reference-conditioned codecs, and includes a mode selector that can fall back to the no-reference path.

The local demo accepts a target screenshot and an optional reference, then shows codec outputs, decoded images, on-disk byte counts, and quality metrics. The neural codecs are lossy. Delta+PNG is the lossless reference-based option. Inputs are fitted and padded to a 512×512 working canvas; the displayed neural bitstreams are decoded from their saved `.bin` files.

## Current status

- The training/evaluation corpus contains 24,000 balanced WebSight-derived screenshot pairs across five change buckets. These are generated modifications, not captured product-version histories; results should be understood in that scope.
- The available No-Reference and Reference-Conditioned checkpoints use one operating point (`lambda=0.0130`) and are stored locally under `checkpoints/`.
- The interactive demo reports actual compressed file bytes. The existing 500-pair pipeline summary reports likelihood-estimated neural bpp; consult its labels and the runbook when interpreting benchmark results.
- Training, checkpoints, datasets, and generated outputs are not included in Git. See [RUNBOOK.md](RUNBOOK.md) and [PROJECT_STATE.md](PROJECT_STATE.md) for the verified project state and detailed operating steps.

## Run the local demo

Use repo-root PowerShell on Windows with Ubuntu/WSL configured, the project’s `.venv_wsl` environment installed, and the epoch-27 checkpoints available locally:

```powershell
.\scripts\wsl\run_demo.ps1
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Upload a PNG, JPEG, or WebP target; add a reference to enable the reference-conditioned and Mode Selection outputs. The demo does inference only and does not train models.

Stop the server with:

```powershell
.\scripts\wsl\stop_demo.ps1
```

For premade comparisons, build the artifacts first and wait for the completion message before opening the demo:

```powershell
.\scripts\wsl\build_premade_demos.ps1
Get-Content .\results\build_premade_demos.log -Wait
```

Detailed setup, verification, troubleshooting, and WSL command guidance are in [RUNBOOK.md](RUNBOOK.md).

## Tests

From repo-root PowerShell, run the Windows-side unit suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp results\pytest --disable-warnings -p no:cacheprovider
```

Some CompressAI-dependent tests are skipped when CompressAI is unavailable in the Windows environment. For codec and checkpoint verification, use the WSL instructions in the runbook; a stub-codec browser test validates the web layer only.

## Repository layout

- `models/` — learned codec architectures
- `scripts/` — training, evaluation, demo, and WSL launch utilities
- `tests/` — unit and integration tests
- `gpu_portable/` — portable training materials
- `website_screenshots/` — demo screenshot pairs and manifest
- `PROJECT_STATE.md` — milestone status and handoff log
- `RUNBOOK.md` — environment and execution instructions

## Contribution code map

See [CONTRIBUTION_CODE_GUIDE.md](CONTRIBUTION_CODE_GUIDE.md) for the essential source files each contributor should study and the support/generated code they can skip.

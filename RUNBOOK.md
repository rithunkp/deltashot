# DeltaShot Runbook

Last audited: 2026-10-22

## Canonical environment

Use WSL2 Ubuntu and `.venv_wsl` for Playwright rendering, dataset generation, CompressAI, and any future training. Verified via `results/wsl_environment_check.txt`: `compressai=1.2.8`, `torch=2.14.0+cpu`, `cuda=False`. The Windows `.venv` remains usable for local unit tests and rendering checks, but it has CPU-only PyTorch and no CompressAI.

## How to run commands in the WSL venv from Windows-side tooling

Always invoke the Linux Python through Ubuntu; do not execute `.venv_wsl/bin/python` directly from Windows. The project scripts below avoid nested shell quoting. For a short environment check:

```powershell
wsl.exe -d Ubuntu -- bash /mnt/c/Rithun/Github/deltashot/scripts/wsl/check_environment.sh
```

The environment script also writes `results/wsl_environment_check.txt`; use that file as the authoritative confirmation when Windows-side output capture is blank.

## Local upload demo (inference only)

The demo page labels the gallery tab **Premade demos**, uses **Modified** as the visible label for the altered test examples, and fits gallery thumbnails inside square frames without cropping or stretching. Restart the local server after HTML edits to reload the page.

Start the CPU-only upload demo from repo-root PowerShell (the launcher waits up to 10 minutes for both checkpoints and their entropy models to load; override with `-StartupTimeoutSeconds` if needed):

```powershell
.\scripts\wsl\run_demo.ps1
```

The launcher converts the resolved Windows repository path using `wslpath -u` after normalizing separators. If WSL path conversion fails, it reports the exit code and captured error rather than continuing with an empty path.

It loads the epoch-27 No-Reference and Reference-Conditioned checkpoints from `checkpoints/`, opens `http://127.0.0.1:8765`, and accepts a PNG/JPEG/WebP target plus an optional reference up to the 16,777,216-pixel limit. Any accepted input is converted to RGB, fitted within 512x512 preserving aspect ratio, then edge-padded to 512x512. The UI shows upload and working dimensions; codec bytes and PSNR are measured on the working image. No training is run. With a reference, the page shows Original, JPEG q75, WebP q75, lossless Delta+PNG, No-Reference, Reference-Conditioned, and Mode Selection. Without a reference, Delta+PNG and Reference-Conditioned are explicitly unavailable and Mode Selection is explicitly skipped.

Startup logs identify the phase reached (Python startup, each checkpoint read, entropy-model update, and server ready). If the timeout expires, check both server logs; a launcher timeout does not itself prove the WSL process exited. Request validation errors return JSON with a specific `error`; unexpected inference errors return HTTP 500 with a specific message and a server-side traceback in `results/local_demo_server.err.log`.

Neural output uses each checkpoint's actual CompressAI `compress()` strings. The demo writes a container `.bin`, reopens that file, and calls `decompress()` using only the saved streams (plus the uploaded reference where required). It checks the restored dimensions, computes PSNR against the uploaded target, and displays both likelihood-estimated bpp and actual file bpp; the difference includes container/mode-header overhead. Each decoded image is saved as a PNG beside its compressed artifact under `results/local_demo_runs/<run-id>/`. Mode Selection minimizes actual file bpp plus `0.013 × RGB MSE`; the 5-byte local mode header is counted in the selected file size. This is the interactive-demo decision rule, distinct from the benchmark's likelihood-estimated selector.

The neural codecs are **lossy**. Delta+PNG is the **lossless path** and requires the same reference at decode time. JPEG/WebP use quality 75. Stop the server with:

```powershell
.\scripts\wsl\stop_demo.ps1
```

Use `-Port 8766` with both scripts to choose another local port. Run records, manifests, `.bin` files and decoded PNGs remain in `results/local_demo_runs/`. Server logs are `results/local_demo_server.log` and `results/local_demo_server.err.log`.

## Real website screenshot samples

`website_screenshots/manifest.json` currently lists 18 demo pairs: six genuine version-to-version captures (Python 3.11/3.12 tutorial, asyncio and typing pages; Bootstrap 5.2/5.3 accordion, navbar and card pages), ten synthetic test-split samples (two in each of the five buckets), and two unrelated public-page pairs. The six version pairs were captured with Playwright Chromium at a 512x512 viewport and device scale factor 1, visually checked, and have no dropped captures. Their measured changed-pixel ratios are in the manifest; they are classified as `major`. These pairs are demo material only, not training or benchmark data. The four legacy standalone samples in `website_screenshots/` remain separate from the paired gallery.

The premade builder creates/refreshes the pair manifest (including version captures if fewer than five are present), copies the balanced synthetic examples, runs all codecs using epoch-27 checkpoints, and writes decoded PNGs and `.bin` files under `results/premade_demos/runs/` plus `results/premade_demos/manifest.json`. It is a WSL/CompressAI job; no training occurs. Run it to completion **before** starting verification; do not overlap multiple verifier runs or the builder with verification.

Mode Selection writes the selected codec ID into its `.bin` container. Decode dispatch must follow that stored ID: a selected No-Reference stream is decoded without a reference tensor, even if the original request included a reference; only the reference-conditioned ID uses the reference-aware decoder. The current `decode_mode_bitstream()` implements this and has stub-based regression tests; CompressAI validation still requires the WSL premade run.

```powershell
.\scripts\wsl\build_premade_demos.ps1
Get-Content results\build_premade_demos.log -Wait
```

For a Windows-only classical preview (no CompressAI required), run `.\.venv\Scripts\python.exe -m scripts.local_demo --classical-only`; it serves the same page and computes JPEG/WebP/Delta+PNG into on-disk artifacts. It does not load neural checkpoints. Choose another port if a WSL demo is already active.

## Local demo verification

Run the detached WSL verifier only after `results/premade_demos/manifest.json` exists. It checks every pair in the gallery through the live endpoint, reference-conditioned `.bin` decoding, invalid/empty input diagnostics, and the two-tab UI controls, then saves `results/demo_verify_live.png` and `results/demo_verify_premade.png`. Reports are `results/demo_verify_report.txt` and `.json`; they are **not** benchmark results. Only the real WSL run uses CompressAI; Windows `--stub-codecs` is web-layer verification and never codec evidence. The verifier stops an incompatible stale/classical-only server and requires `neural_available: true` before accepting the endpoint. The last local stub pass covered all 18 gallery pairs and the two browser tabs, with 24 checks passing; it did not exercise the trained checkpoints.

```powershell
.\scripts\wsl\verify_demo.ps1
Get-Content results\demo_verify.log -Wait
```

When the detached run finishes, require JSON `stub_codecs: false` and `status: PASS`; paste `results/demo_verify_report.txt` and `.json`. If it fails, also provide `results/demo_verify.log` and `results/local_demo_server.err.log`. WSL stdout/stderr captures are `results/demo_verify_wsl_stdout.log` and `_stderr.log`.

Run these one at a time from repo-root PowerShell. Start the builder in one terminal, and follow its log in a second terminal. After the log prints `wrote 18 premade comparisons`, press Ctrl+C in the log-following terminal (this only stops tailing), then confirm the manifest before starting the verifier. The build launcher creates/clears its log immediately to avoid the initial missing-file race:

```powershell
.\scripts\wsl\stop_demo.ps1
.\scripts\wsl\build_premade_demos.ps1
# In another terminal:
Get-Content .\results\build_premade_demos.log -Wait
# After the final wrote line, Ctrl+C the log follower, then:
Test-Path .\results\premade_demos\manifest.json
.\scripts\wsl\run_demo.ps1
.\scripts\wsl\verify_demo.ps1
Get-Content .\results\demo_verify.log -Wait
```

`run_demo.ps1` reports ready only when `/health` confirms API v3 and `neural_available: true`; it stops/replaces an incompatible classical-only listener rather than labeling it the epoch-27 neural server. `verify_demo.ps1` refuses to launch before the premade manifest exists and reports the two actual screenshot paths.

For a Windows-only web-layer check (no CompressAI and no checkpoint inference), run:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_demo --stub-codecs --port 8767 --startup-timeout 60
```

It writes the same report paths and browser screenshots, but the report must say `stub_codecs: true`; these outputs are not real neural codec or model-quality results.

For any run expected to exceed a few seconds, detach the **Windows `wsl.exe` process**, keeping the command itself in the foreground inside the distro. This keeps Ubuntu alive while the command runs. `launch_detached.sh` (which backgrounds a child inside WSL) has proved unreliable when Ubuntu shuts down after the caller exits. Use `cmd.exe` to start a hidden independent WSL process, and redirect output inside WSL with `run_logged.sh`:

```powershell
cmd.exe /c start "" /b wsl.exe -d Ubuntu -- bash /mnt/c/Rithun/Github/deltashot/scripts/wsl/run_logged.sh JOB .venv_wsl/bin/python -u -m PACKAGE.MODULE ARGS
Get-Content results/JOB.log -Tail 30
```

For a resumable baseline, use its `--resume` flag and retain the same JSONL output path. `run_compressai.py` writes one JSON line per `(model, quality)` configuration; a completed line contains its full per-image `results` array, so one line can represent all 2,000 test images.

## Manual steps: previous WSL launchers

These launchers remain available for a user-authorized WSL Phase 3 workflow. They were not executed in the authoring session because WSL returned `E_ACCESSDENIED`; do not infer any fresh throughput or training result from their presence.

```powershell
.\scripts\wsl\run_tests.ps1
Get-Content results/test_output.txt -Wait
.\scripts\wsl\run_throughput_probe.ps1
Get-Content results/throughput_probe.log -Wait
```

Inspect `results/throughput_probe_metrics.json` before choosing a Phase 3 scope. `PairsPerEpoch` is the total and must be divisible by five:

```powershell
.\scripts\wsl\run_phase3_train.ps1 -Lambda 0.013 -Epochs 30 -PairsPerEpoch 20000
.\scripts\wsl\run_phase3_train.ps1 -Lambda 0.013 -Epochs 30 -PairsPerEpoch 20000 -Resume
```

## Matched-epoch evaluation

Use this after the selected zero-based epoch checkpoint exists for both codecs and the final No-Reference epoch 27 checkpoint is present. `K=9` means 10 epochs completed. The default evaluation is 50 stratified test pairs per bucket, joined to the existing classical baseline records by the exact same pair IDs. This evaluates likelihood-estimated neural bpp, not actual encoded bitstream size. The job uses CPU and competes with an active training process, so keep the subset small and run only when the contention is acceptable.

```powershell
.\scripts\wsl\run_matched_eval.ps1 -Epoch 9 -PerBucket 50
Get-Content results\matched_eval_epoch_9.log -Tail 30 -Wait
```

Optional checkpoint overrides are repo-relative paths (or WSL paths): `-NoRefCheckpoint`, `-RefCheckpoint`, and `-LongerNoRefCheckpoint`. Outputs are isolated as `results/matched_epoch_K_eval.jsonl`, `matched_epoch_K_modes.jsonl`, `matched_epoch_K_summary.json`, plus the matched-epoch section in `results/demo.html`. Require all five buckets, the same pair IDs in classical rows, and inspect `mode_selection.unrelated_correctness` in the summary; the tool does not assert that an empirical PASS exists.

## RUN ORDER

Run from a normal PowerShell window at `C:\Rithun\Github\deltashot`. The first cloud attempt failed in its PowerShell-quoted bootstrap before downloading the bundle or entering the job script. The replacement launcher uses `HfApi.run_job` with an argv list; it has only been syntax-checked locally and not submitted. For laptop execution, the detached WSL launcher keeps the Windows-side `wsl.exe` alive.

1. Run preflight first (five-step smoke for both codecs, then an 8-pair evaluation). Add -SkipPytest if pytest already passed; this skips only pytest and still runs both codec smokes and the combined evaluation. A full PASS requires noref=PASS, reference_conditioned=PASS, eval=PASS, and results/preflight/eval.jsonl containing both noref and reference_conditioned fields. On failure, pipeline_status.txt points to the full subprocess output/traceback log:

```powershell
.\scripts\wsl\run_pipeline.ps1 -Stages preflight -SkipPytest
Get-Content results/pipeline_status.txt -Wait
```

The launcher creates `pipeline_status.txt` before detaching and starts `wsl.exe` directly as a hidden foreground host process. Without `-SkipPytest`, follow `results/pipeline_pytest.txt`. WSL startup errors are captured separately in `results/pipeline_wsl_stderr.log` (standard output in `pipeline_wsl_stdout.log`).

2. For Hugging Face, install the Python client into the active PowerShell Python environment, set `HF_TOKEN` without putting its value in command history, and upload the private code/data bundle. The bundle contains runnable code plus 10,000 balanced training pairs (2,000/bucket), 500 validation pairs (100/bucket), and 500 test pairs (100/bucket):

```powershell
python -m pip install -U huggingface_hub
python -m scripts.hf.upload_bundle USERNAME/deltashot-bundle
```

3. Run the CPU plumbing dryrun first. It downloads the bundle, installs CompressAI/runtime dependencies, creates the private Model artifact repo before any checkpoint upload, runs both five-step smoke trainings and the 8-pair evaluation with `device=auto`, and does not run pytest or calibration. Its EXIT trap uploads logs and partial artifacts on failure. The bundle download was observed to take nearly 15 minutes for 701 MB / 33,059 files, so dryrun has a 60-minute timeout to leave room for setup and smoke checks; on `cpu-upgrade` the maximum cost estimate is about $0.03. Type `YES` at the confirmation prompt:

```powershell
./scripts/hf/launch_job.ps1 -BundleRepo USERNAME/deltashot-bundle -ArtifactRepo USERNAME/deltashot-results -Mode dryrun -Flavor cpu-upgrade
```

Wait for completion in the Hugging Face Jobs page, then download results with `hf download USERNAME/deltashot-results --repo-type model --local-dir ./hf_results`.

Review `hf_results/results/pipeline_status.txt`, `hf_results/results/preflight.json`, and all rows of `hf_results/results/preflight/eval.jsonl`. Require both codec smoke statuses PASS and both `noref` and `reference_conditioned` fields in the evaluation rows before moving on.

4. Run calibration on T4 only after dryrun passes. This executes pytest, both smokes, an 8-pair evaluation, and timed calibration, but not training. Timeout is 15 minutes; maximum cost estimate at $0.40/hour is $0.10. Type `YES`:

```powershell
./scripts/hf/launch_job.ps1 -BundleRepo USERNAME/deltashot-bundle -ArtifactRepo USERNAME/deltashot-results -Mode calibrate -Flavor t4-small
```

Wait for job completion and download with the same `hf download` command. Review `hf_results/results/pipeline_calibration.json` and `hf_results/results/pipeline_status.txt`; confirm the proposed crop size, pairs/epoch, epochs, and budget before launching training.

5. Launch the bounded full cloud pipeline only after reviewing calibration. It uses one lambda (0.013), no AMP, epoch checkpoints, and uploads checkpoints/metrics after each epoch. Timeout is 4.5 hours; maximum cost estimate at $0.40/hour is $1.80. Type `YES`:

```powershell
./scripts/hf/launch_job.ps1 -BundleRepo USERNAME/deltashot-bundle -ArtifactRepo USERNAME/deltashot-results -Mode full -Flavor t4-small
```

Wait for job completion, then download the final artifacts with `hf download USERNAME/deltashot-results --repo-type model --local-dir ./hf_results`.

6. Laptop overnight training uses the resumable pipeline with an approximately 8-hour pipeline budget. Calibration sizes the work from measured throughput and may reduce pairs/resolution/epochs; this is a budget target, not a strict wall-clock kill timer. If you already ran the standalone preflight above, pass `-Resume` so its PASS result is reused. Results at reduced settings must be labeled reduced-scale:

```powershell
.\scripts\wsl\run_pipeline.ps1 -BudgetHours 8 -Resume
Get-Content results/pipeline_status.txt -Wait
```

To request a safe pause, let the current epoch finish and checkpoint before exit:

```powershell
.\scripts\wsl\stop_pipeline.ps1
Get-Content results/pipeline_status.txt -Wait
```

Wait for `train_noref | PAUSED` or `train_ref | PAUSED` in the status file and for `results/pipeline.log` to report the pipeline stopped. Then continue the same calibrated run:

```powershell
.\scripts\wsl\run_pipeline.ps1 -BudgetHours 8 -Resume
Get-Content results/pipeline_status.txt -Wait
```

The resume path restores model/optimizer plus Python, NumPy, Torch, and DataLoader RNG state, and starts after the last completed epoch checkpoint. Do not start a second pipeline while the first is still exiting. Without `-Resume`, stage status is reset and preflight/calibration run fresh. The cloud entrypoint restores existing checkpoints/results from the private artifact repo before a resumed full run, and its EXIT trap uploads logs plus partial results/checkpoints after success or failure. The launcher submits through the installed `huggingface_hub.HfApi.run_job` Python API with a list-valued command; no Hugging Face CLI shell command is involved. API signature was inspected locally, but no Job was submitted or run in this session.

## Verified dataset

`data/synthetic_pairs_full/` is complete. It contains 24,000 512x512 reference/target pairs:

| Split | Total | Identical | Minor | Moderate | Major | Unrelated |
|---|---:|---:|---:|---:|---:|---:|
| Train | 20,000 | 4,000 | 4,000 | 4,000 | 4,000 | 4,000 |
| Validation | 2,000 | 400 | 400 | 400 | 400 | 400 |
| Test | 2,000 | 400 | 400 | 400 | 400 | 400 |

Do not regenerate this corpus. The full build required about 30 hours. Representative test pairs were visually checked on 2026-09-08 and their image changes matched the metadata labels.

**Data provenance:** WebSight pages are synthetically generated, and this project further mutates/renders those pages to create screenshot pairs. These are not captured real product histories. Label the dataset and all results as synthetic WebSight-derived mutation data; do not imply deployment-level generalization without real-history validation.

## Existing artifacts

- `results/baselines_classical_full_test.jsonl`: completed full 2,000-pair test-set JPEG/WebP/Delta+PNG evaluation, exactly 400 records per degree-of-change bucket. Delta+PNG is lossless for all 2,000 targets.
- `results/baselines_classical.jsonl`: legacy 24-pair-scale classical run only (20 records).
- `results/baselines_compressai.jsonl`: legacy 24-pair-scale generic CompressAI run only (8 records).
- `checkpoints/pipeline_noref_lambda_00130/` and `checkpoints/pipeline_reference_lambda_00130/`: user-confirmed 28-epoch CPU training runs at 128px crops, batch 4, lambda 0.0130; epoch 0–27 checkpoints exist for both. This is one operating point, not a multi-point RD curve.
- `results/baselines_compressai_full_test.jsonl`: completed full 2,000-pair generic pretrained `bmshj2018-hyperprior` evaluations at q=1, 3, 5, and 7. Each configuration line has exactly 2,000 per-image results; quality 3 was retained, not rerun.
- `checkpoints/poc_1000_64_lambda_00130/` and `results/poc_training_metrics.json`: complete bounded CPU proof-of-concept: 1,000 balanced train pairs, 64px crops, batch 8, lambda 0.0130, five epochs. Loss fell from 830.994 to 20.778 over 65 logged samples. `results/poc_examples/` contains one real reference/target/reconstruction triplet per bucket. This remains architecture validation only, not a final or converged model.
- User-provided `results/pipeline_status.txt` confirms No-Reference and Reference-Conditioned PASS at lambda 0.0130, crop 128, 20,000 pairs/epoch, 28 epochs, CPU, AMP off. It also reports evaluation PASS on 500 pairs (100/bucket) and unrelated-bucket Mode Selection PASS. This is subset evidence with likelihood-estimated bpp, not full 2,000-pair or actual-bitstream benchmark evidence.
- `scripts/wsl/run_throughput_probe.ps1`, `run_phase3_train.ps1`, and `run_tests.ps1`: user-run detached launchers documented above. Not executed here due the WSL sandbox denial.
- `gpu_portable/`: self-contained No-Reference trainer and README plus `data/synthetic_pairs_subset.zip` (1,000 train pairs, 200/bucket; 200 validation pairs, 40/bucket). This is a small portability sample, not a final training set.
- `models/reference_encoder.py`, `models/alignment.py`, and `models/conditional_codec.py`: Phase 4 v1 uses simple feature concatenation; cross-attention v2 remains unstarted.
- `results/demo.html`: standalone review page regenerated from present artifacts on 2026-10-15. Confirm generic/Poc sections and chart are displayed correctly before presenting; page generation had no recorded current test-suite rerun.
- The new upload demo is `scripts/local_demo.html`, served by `scripts/local_demo.py`. It stores actual `.bin` files and decoded PNGs under `results/local_demo_runs/`. Checkpoint-backed 512x512 inference is pending verification.
- No full 2,000-pair learned-codec evaluation, multi-point RD curve, or BD-rate-by-bucket result is established yet.

## Current order of work

1. Phase 2 classical and generic pretrained baselines are complete on the 2,000-pair test set. Phase 3/4 are complete at one CPU operating point (lambda 0.0130, 128px crops, batch 4, 28 epochs each), per user-provided logs.
2. The user-provided learned-codec evaluation covered 500 test pairs (100/bucket); Mode Selection reports unrelated-bucket PASS. It is subset evidence with likelihood-estimated bpp, not a full actual-bitstream benchmark.
3. Verify the local upload demo using epoch-27 checkpoints and a 512x512 target, with and without a reference. Then run the full 2,000-pair learned-codec evaluation and Mode Selection before treating Phase 6 as fully validated.
4. Add matched operating points only if compute allows. A single lambda cannot establish an RD curve or BD-rate. Defer cross-attention; clearly disclose synthetic data provenance and the absence of real-history validation.

## Resource plan and expected timeline

Use this as a planning range, not a promise. The first matched lambda is complete according to the user's pipeline logs: 20,000 pairs/epoch, 128px crops, batch 4, 28 epochs, lambda 0.0130, CPU. The older 20-step 256px throughput estimate is unreliable and must not be presented as measured timing. Additional-lambda training, full-test evaluation, and benchmark timing remain unmeasured.

- **Minimum credible synthetic-data v1:** the first matched-lambda training and 500-pair subset Mode Selection check are reported complete. Remaining work is checkpoint-backed demo verification and a full 2,000-pair evaluation, whose runtime is not measured here.
- **Small multi-point benchmark:** for each additional lambda, budget about 8 hours for No-Reference plus 8-16 hours for reference-conditioned training, then evaluate. Three or four total lambda points therefore imply roughly 45-90 additional training hours beyond the current mostly-complete No-Reference point, plus evaluation. On a single CPU pipeline this is several continuous days; realistically allow about 1-2 calendar weeks including interruptions, analysis, and report/demo polish.
- **Recommended priorities:** (1) verify the local demo at 512x512; (2) validate all 2,000 test pairs and the unrelated-bucket Mode Selection bar; (3) add matched lambda points only if compute remains. Keep cross-attention and real-world collection as future work for this constrained delivery.

If only the first lambda finishes, report actual per-bucket bpp/PSNR/MS-SSIM and rate comparisons at that operating point, and mark the RD curve/BD-rate work incomplete. Never infer BD-rate from one point or present partial epochs as trained results.

## Current orchestration notes

- `scripts/run_pipeline.py` runs preflight, calibration, No-Reference training, v1 reference-conditioned training, evaluation, Mode Selection, and report stages in order; use `--stages` only to intentionally select stages. `--resume` retains passed stages and trainer epoch checkpoints. `scripts/wsl/stop_pipeline.ps1` requests a safe epoch-boundary pause through `results/STOP_AFTER_EPOCH`; resuming clears it. Missing reference checkpoints make Mode Selection `SKIP` with an explicit note that the unrelated correctness criterion was not tested, never `PASS`.
- Device resolution supports `auto`, `cpu`, and `cuda`; training uses FP32 (AMP off) for entropy-model numerical safety. Checkpoint resume restores Python/NumPy/Torch/DataLoader RNG state. `HF_TOKEN` plus `DELTASHOT_ARTIFACT_REPO` enables epoch-by-epoch checkpoint and metrics uploads.
- Phase 4 v1 training is user-confirmed complete; the local demo's CompressAI round-trip remains untested in Windows because CompressAI is absent there and WSL launch returned `E_ACCESSDENIED`. No actual-bitstream benchmark is claimed by the likelihood-based pipeline output.


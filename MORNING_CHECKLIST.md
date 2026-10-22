# DeltaShot Morning Checklist

## Open First

1. Open `results/pipeline_status.txt`. Every stage records PASS, FAIL, or SKIP and a reason. A missing file means the pipeline has not started.
2. Open `results/pipeline_calibration.json`. Confirm the actual device, measured seconds/step, crop size, pairs/epoch, and epoch count before treating the chosen scale as intentional.
3. If evaluation ran, open `results/pipeline_summary.json`. `mode_selection.unrelated_correctness` must be PASS before claiming fallback correctness. FAIL means stop and inspect the unrelated bucket; missing reference results mean the check was not evaluated.
4. Open `results/demo.html` for the per-bucket comparison, loss samples, and saved reconstructions. It labels likelihood bpp as estimated and below-spec runs as REDUCED-SCALE.
5. For a cloud run, check the private artifact repo for `checkpoints/` and `results/`; checkpoints and metrics are synchronized at each completed epoch and again at job exit.

## Outcome Guide

- `preflight PASS`: pytest, No-Reference smoke training, and 8-pair evaluation passed. If the reason mentions reference smoke failure, Phase 4 training was intentionally skipped and its compute was allocated to No-Reference.
- `calibrate PASS`: use its observed rates and chosen settings; calibration does not train a reportable model.
- `train_noref PASS` / `train_ref PASS`: epoch checkpoint and metrics exist. Verify pair count, crop, epochs, lambda, and device in the metric JSON.
- `evaluate PASS`: records are from the test split, contain degree labels, and report likelihood-derived estimated bpp, PSNR, MS-SSIM, plus same-pair classical records.
- `mode_selection PASS`: the unrelated-bucket per-pair J check passed with the one-bit cost included when the reference mode is selected. This is a single-lambda correctness check, not BD-rate evidence.
- Any FAIL: inspect the named log in `results/`; do not present a failed or partial stage as a positive result. `SKIP` is not a pass.

## Manual Run Order

See the **RUN ORDER** section in `RUNBOOK.md` for exact commands and the laptop/Hugging Face sequence. The HF launcher verifies `hf jobs run --help` and `hf jobs hardware` locally before prompting to submit. The launcher was not exercised in the authoring environment because the `hf` CLI was unavailable there.

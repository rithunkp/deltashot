
## Rithun K P — Model and training

**Core files**

- `models/baseline_codec.py` — constructs the No-Reference control from CompressAI's `ScaleHyperprior`.
- `models/conditional_codec.py` — Reference-Conditioned v1: reference features, concatenation/fusion, entropy model, and actual `compress()`/`decompress()` paths. Its `load_state_dict()` contains the CompressAI dynamic-buffer loading fix.
- `models/reference_encoder.py` and `models/alignment.py` — shared multi-scale reference features and v1 resize/project/concatenate operation.
- `scripts/train_baseline.py` — No-Reference training loop, rate-distortion loss, optimizer, checkpoint contents, and `--resume` behavior.
- `scripts/train_reference.py` — matched Reference-Conditioned training loop and checkpoint resume.

**Useful tests**

- `tests/test_reference_conditioning.py`
- `tests/test_checkpoint_roundtrip.py`

**Be ready to explain**

- Both models use the same base `ScaleHyperprior` configuration, training data settings, seed, optimizer, learning rate (`1e-4`), and lambda (`0.0130`). The control isolates the effect of adding reference information.
- The loss is CompressAI `RateDistortionLoss(metric="mse")`: estimated rate plus lambda-weighted distortion. Lambda trades compressed rate against reconstruction error.
- GDN, the base analysis/synthesis transforms, and hyperprior transforms come from CompressAI's `ScaleHyperprior`; they are not handwritten layer-by-layer in this repo. The custom reference-conditioned fusion layers are in `models/conditional_codec.py`.

## Balagopal V — Data and baselines

**Core files**

- `scripts/fetch_websight_pages.py` — obtains the source WebSight page records.
- `data/rendering.py` — renders HTML pages into screenshot images.
- `data/mutations/page.py` — shared HTML parsing and mutation helpers.
- `data/mutations/color.py`, `insert.py`, `layout.py`, `remove.py`, `replace.py`, `text.py` — the six mutation operators.
- `data/bucketing.py` — changed-pixel ratio and degree-of-change bucket labels.
- `scripts/generate_balanced_synthetic_pairs.py` and `data/dataset.py` — balanced pair generation and training/evaluation loading.
- `scripts/baselines/jpeg_webp.py`, `delta_png.py`, and `run_classical.py` — JPEG, WebP, and lossless reference-delta implementations and runs.
- `scripts/baselines/run_compressai.py` — generic pretrained CompressAI baseline.

**Useful tests**

- `tests/test_mutations.py`
- `tests/test_balanced_generation.py`
- `tests/test_baselines.py`

**Be ready to explain**

- What is an image pair, how the five buckets are assigned, and how each baseline encodes/decodes.
- The WebSight-derived screenshots are modified/generated pairs, not real product-version histories. Explain the synthetic-data limitation and how split/bucket separation is handled; do not describe them as real-world change histories.

## Saavan Rajeev — Evaluation, Mode Selection, and demo

**Core files**

- `scripts/evaluate_codecs.py` — evaluates reconstructions and computes bpp estimate, PSNR, and MS-SSIM.
- `scripts/select_modes.py` — applies the evaluation-time per-pair Mode Selection rule and summarizes results by bucket.
- `scripts/mode_decision.py` — live-demo selector using actual encoded bytes and distortion.
- `scripts/bitstream.py` — writes/reads neural and mode containers so decoding can use the saved bitstreams.
- `scripts/local_demo.py` and `scripts/local_demo.html` — upload flow, real encoding/decoding, displayed byte counts/PSNR, and browser UI.
- `slide_assets/build_slide_assets.py` — creates the evaluation charts from saved result files.
- `scripts/build_pipeline_report.py` — generates the static report page, if discussing that page as well as the live demo.

**Useful tests**

- `tests/test_mode_decision.py`
- `tests/test_bitstream.py`
- `tests/test_local_demo_roundtrip.py`

**Be ready to explain**

- bpp (bits per pixel), PSNR (pixel-level reconstruction error in dB), and MS-SSIM (structural similarity); distinguish likelihood-estimated bpp in evaluation from actual on-disk byte counts in the live demo.
- Mode Selection compares rate-distortion costs and can fall back to No-Reference when the reference path is not beneficial. The unrelated-bucket check is a required correctness check.
- The neural codecs are lossy. Delta+PNG is the separate lossless reference-based path. State the dataset/training scope and single-lambda limitation when presenting results.

## Skip for the contribution presentation

Unless asked specifically about operations or deployment, you do not need to study `scripts/wsl/`, `scripts/hf/`, `scripts/runtime.py`, `scripts/verify_environment.py`, environment audits, old POC exporters, or generated files under `results/`, `checkpoints/`, `data/`, and `slide_assets/`. Those are launch/support code or artifacts, not the central implementation contributions. For model layer details supplied by CompressAI, identify the library and explain its role rather than presenting dependency internals as code written in this project.

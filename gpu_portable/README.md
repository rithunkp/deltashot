# DeltaShot GPU Training Bundle

This bundle contains a self-contained No-Reference ScaleHyperprior trainer and a small balanced data sample. It trains the Phase 3 baseline only; it does not train reference conditioning or Mode Selection.

## Run in Colab or Kaggle

1. Enable a GPU runtime and install dependencies:

```bash
pip install torch torchvision compressai pillow
```

2. Extract `data/synthetic_pairs_subset.zip` so it creates `data/synthetic_pairs_subset/{train,val}/pair_*/`.
3. Run from the directory containing `train.py`:

```bash
python train.py --dataset data/synthetic_pairs_subset --epochs 30 --batch-size 4 --patch-size 256 --lambda-value 0.013 --pairs-per-bucket 200 --save-dir checkpoints/lambda_00130 --metrics results/lambda_00130_metrics.json
```

Resume after interruption with the same arguments and add `--resume`. Checkpoints are saved after each epoch; resume starts at the next epoch and restores optimizer state. Keep the same save directory and configuration when resuming. Metrics are written after each completed epoch. The bundled training subset has 200 pairs per bucket; the validation split has 40 pairs per bucket and is provided for future evaluation (this script does not evaluate validation yet).

The bundled sample is for portability and smoke-scale GPU runs, not final research numbers. Record the actual pair count, patch size, epochs, lambda, and device with any reported result.

"""Evaluate a pretrained CompressAI generic learned codec on DeltaShot synthetic pairs.

This is baseline #3 from Section 6.1: a pretrained natural-image codec evaluated
as-is on screenshots, to show the domain-specialization gap that this project's
own No-Reference Codec aims to close.

Output is one JSON record per (pair, quality) with the rate-distortion
characteristics of the pretrained model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

try:
    from compressai.zoo import bmshj2018_hyperprior, mbt2018_mean
except ImportError:
    print("compressai not installed. Run scripts under WSL2 venv.", file=sys.stderr)
    raise

from scripts.baselines.data_loader import load_image, load_pairs
from scripts.baselines.metrics import bits_per_pixel, psnr


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a [1, 3, H, W] float tensor in [0, 1]."""
    arr = torch.from_numpy(__import__("numpy").asarray(image, dtype="float32") / 255.0)
    return arr.permute(2, 0, 1).unsqueeze(0)


def to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] float tensor in [0, 1] back to a PIL image."""
    import numpy as np
    arr = tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).round().astype("uint8"), mode="RGB")


def evaluate_model(
    model_name: str,
    quality: int,
    pairs: list,
    device: str = "cpu",
) -> dict:
    """Run a single CompressAI pretrained model at a given quality on a set of pairs."""
    if model_name == "bmshj2018-hyperprior":
        model = bmshj2018_hyperprior(quality=quality, pretrained=True)
    elif model_name == "mbt2018-mean":
        model = mbt2018_mean(quality=quality, pretrained=True)
    else:
        raise ValueError(f"unknown compressai model: {model_name}")
    model = model.to(device).eval()

    results = []
    for pair in pairs:
        target = load_image(pair.target_path)
        with torch.no_grad():
            x = to_tensor(target).to(device)
            out_enc = model.compress(x)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            recon = to_image(out_dec["x_hat"])

        bpp = sum(
            sum(len(s) for s in strs) * 8
            for strs in out_enc["strings"]
        ) / (target.size[0] * target.size[1])
        p = psnr(target, recon)
        results.append(
            {
                "pair_id": pair.pair_id,
                "split": pair.split,
                "degree_of_change": pair.degree_of_change,
                "bpp": bpp,
                "psnr_db": p,
            }
        )

    return {
        "model": model_name,
        "quality": quality,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pretrained CompressAI codec as a baseline.")
    parser.add_argument("--input", type=Path, default=Path("data/synthetic_pairs"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--models", nargs="+", default=["bmshj2018-hyperprior", "mbt2018-mean"]
    )
    parser.add_argument("--qualities", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/baselines_compressai.jsonl"))
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip model/quality configurations already complete in an existing JSONL output.",
    )
    args = parser.parse_args()

    pairs = load_pairs(args.input, split=args.split)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(record.get("results", [])) == len(pairs):
                completed.add((record.get("model"), record.get("quality")))

    with open(args.output, "a" if args.resume else "w", encoding="utf-8") as f:
        for model_name in args.models:
            for quality in args.qualities:
                if (model_name, quality) in completed:
                    print(f"skipping completed {model_name} q={quality}", flush=True)
                    continue
                print(f"running {model_name} q={quality} on {len(pairs)} pairs...", flush=True)
                record = evaluate_model(model_name, quality, pairs, device=args.device)
                f.write(json.dumps(record) + "\n")
                f.flush()


if __name__ == "__main__":
    main()

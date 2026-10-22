"""Run the JPEG and WebP classical-codec baselines on a synthetic pair directory.

For each (reference, target) pair in the given split directory, this script:
1. Loads the target image.
2. Compresses it at a sweep of JPEG qualities.
3. Compresses it at a sweep of WebP qualities.
4. Also runs the naive delta+PNG baseline using the reference.
5. Computes PSNR vs. the original target for each quality.
6. Writes a per-pair JSON record with the rate-distortion curve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from scripts.baselines.delta_png import compress_delta_png, decompress_delta_png
from scripts.baselines.jpeg_webp import (
    compress_jpeg,
    compress_webp,
    decompress_jpeg,
    decompress_webp,
)
from scripts.baselines.metrics import bits_per_pixel, psnr


def evaluate_pair(
    reference_path: Path, target_path: Path, qualities: list[int]
) -> dict:
    """Run JPEG, WebP, and delta+PNG on a single (reference, target) pair."""
    with Image.open(reference_path) as ref_img, Image.open(target_path) as tgt_img:
        ref = ref_img.convert("RGB")
        tgt = tgt_img.convert("RGB")

        jpeg_curve = []
        for q in qualities:
            comp = compress_jpeg(tgt, q)
            recon = decompress_jpeg(comp.buffer)
            recon = recon.resize(tgt.size, Image.Resampling.BICUBIC)
            jpeg_curve.append(
                {
                    "quality": q,
                    "bits_per_pixel": bits_per_pixel(
                        comp.compressed_size_bytes, tgt.size[0], tgt.size[1]
                    ),
                    "psnr_db": psnr(tgt, recon),
                    "compressed_bytes": comp.compressed_size_bytes,
                }
            )

        webp_curve = []
        for q in qualities:
            comp = compress_webp(tgt, q)
            recon = decompress_webp(comp.buffer)
            recon = recon.resize(tgt.size, Image.Resampling.BICUBIC)
            webp_curve.append(
                {
                    "quality": q,
                    "bits_per_pixel": bits_per_pixel(
                        comp.compressed_size_bytes, tgt.size[0], tgt.size[1]
                    ),
                    "psnr_db": psnr(tgt, recon),
                    "compressed_bytes": comp.compressed_size_bytes,
                }
            )

        delta_result = compress_delta_png(ref, tgt)
        delta_recon = decompress_delta_png(ref, delta_result)
        delta_record = {
            "bits_per_pixel": bits_per_pixel(
                delta_result.delta_size_bytes, tgt.size[0], tgt.size[1]
            ),
            "psnr_db": psnr(tgt, delta_recon),
            "compressed_bytes": delta_result.delta_size_bytes,
        }

        with open(target_path.with_name("meta.json")) as f:
            meta = json.load(f)

    return {
        "pair_id": target_path.parent.name,
        "split": target_path.parent.parent.name,
        "degree_of_change": meta.get("degree_of_change", "unknown"),
        "jpeg": jpeg_curve,
        "webp": webp_curve,
        "delta_png": delta_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JPEG/WebP/delta+PNG baselines on a synthetic-pair split.")
    parser.add_argument(
        "--input", type=Path, default=Path("data/synthetic_pairs"), help="Root of synthetic pairs tree (with train/val/test subdirs)."
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--qualities", type=int, nargs="+", default=[10, 25, 50, 75, 90]
    )
    parser.add_argument("--output", type=Path, default=Path("results/baselines_classical.jsonl"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only pairs absent from an existing JSONL output file.",
    )
    args = parser.parse_args()

    split_dir = args.input / args.split
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found: {split_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pair_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())

    completed_ids = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                completed_ids.add(json.loads(line)["pair_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    mode = "a" if args.resume else "w"
    processed = len(completed_ids)
    with open(args.output, mode, encoding="utf-8") as out_f:
        for pair_dir in pair_dirs:
            if pair_dir.name in completed_ids:
                continue
            ref = pair_dir / "reference.png"
            tgt = pair_dir / "target.png"
            if not (ref.exists() and tgt.exists()):
                continue
            try:
                record = evaluate_pair(ref, tgt, args.qualities)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                processed += 1
                if processed % 25 == 0:
                    print(f"processed {processed}/{len(pair_dirs)}", flush=True)
            except Exception as e:
                print(f"skip {pair_dir.name}: {e}")


if __name__ == "__main__":
    main()

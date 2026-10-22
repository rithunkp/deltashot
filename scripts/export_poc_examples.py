"""Export real No-Reference proof-of-concept reconstructions for the demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from models.baseline_codec import get_baseline_codec


BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_pairs_full"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/poc_examples"))
    parser.add_argument("--patch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = get_baseline_codec(quality=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.update(force=True)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen = set()
    for pair_dir in sorted((args.dataset / "test").iterdir()):
        meta = json.loads((pair_dir / "meta.json").read_text(encoding="utf-8"))
        bucket = meta["degree_of_change"]
        if bucket not in BUCKETS or bucket in seen:
            continue
        seen.add(bucket)
        pair_output = args.output / bucket
        pair_output.mkdir(exist_ok=True)
        reference = Image.open(pair_dir / "reference.png").convert("RGB")
        target = Image.open(pair_dir / "target.png").convert("RGB")
        reference = TF.center_crop(reference, [args.patch_size, args.patch_size])
        target = TF.center_crop(target, [args.patch_size, args.patch_size])
        with torch.no_grad():
            reconstruction = tensor_to_image(model(TF.to_tensor(target).unsqueeze(0))["x_hat"])
        reference.save(pair_output / "reference.png")
        target.save(pair_output / "target.png")
        reconstruction.save(pair_output / "reconstruction.png")
        manifest.append({"bucket": bucket, "pair_id": pair_dir.name, "meta": meta})
        if len(seen) == len(BUCKETS):
            break
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

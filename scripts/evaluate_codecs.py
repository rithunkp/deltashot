"""Evaluate trained codecs on a deterministic, stratified held-out test subset."""
import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from pytorch_msssim import ms_ssim

from data.dataset import ScreenshotDataset
from models.baseline_codec import get_baseline_codec
from models.conditional_codec import ReferenceConditionedCodec
from scripts.matched_epoch import resolve_matched_checkpoints
from scripts.runtime import resolve_device


def load_pairs(root, limit_per_bucket=None, limit_total=None, seed=24041):
    pairs = []
    for pair_dir in sorted((Path(root) / "test").iterdir()):
        if not pair_dir.is_dir() or not (pair_dir / "meta.json").is_file():
            continue
        meta = json.loads((pair_dir / "meta.json").read_text(encoding="utf-8"))
        pairs.append({"pair_dir": pair_dir, "pair_id": pair_dir.name,
                      "degree_of_change": meta.get("degree_of_change", "unknown")})
    if limit_per_bucket is not None:
        import random
        rng = random.Random(seed)
        selected = []
        for bucket in ("identical", "minor", "moderate", "major", "unrelated"):
            group = [pair for pair in pairs if pair["degree_of_change"] == bucket]
            selected.extend(rng.sample(group, min(limit_per_bucket, len(group))))
        pairs = sorted(selected, key=lambda item: item["pair_id"])
    if limit_total is not None:
        pairs = pairs[:limit_total]
    return pairs


def tensor_from_image(path, device):
    import numpy as np
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device), image


def score(output, target):
    likelihoods = output["likelihoods"]
    pixels = target.shape[-2] * target.shape[-1]
    rate_bits = sum(torch.log(likelihood).sum() for likelihood in likelihoods.values())
    estimated_bpp = float((-rate_bits / math.log(2.0) / pixels).item())
    reconstruction = output["x_hat"].clamp(0, 1)[..., :target.shape[-2], :target.shape[-1]]
    mse = float(torch.mean((reconstruction - target) ** 2).item())
    psnr = None if mse == 0 else 10.0 * math.log10(1.0 / mse)
    similarity = float(ms_ssim(reconstruction, target, data_range=1.0, size_average=True).item())
    return {"estimated_bpp_from_likelihoods": estimated_bpp, "mse": mse,
            "psnr_db": psnr, "ms_ssim": similarity}, reconstruction


def restore(model, checkpoint_path, device, expected_epoch=None):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if expected_epoch is not None and state.get("epoch") != expected_epoch:
        raise ValueError(f"{checkpoint_path} stores epoch {state.get('epoch')}; expected {expected_epoch}")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_pairs_full"))
    parser.add_argument("--epoch", type=int, help="Zero-based matched checkpoint epoch; epoch 9 means 10 epochs completed")
    parser.add_argument("--noref-checkpoint", type=Path)
    parser.add_argument("--ref-checkpoint", type=Path, default=None)
    parser.add_argument("--longer-noref-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lambda-value", type=float, default=0.013)
    parser.add_argument("--limit-per-bucket", type=int, default=None)
    parser.add_argument("--limit-total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=24041)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reconstruction-dir", type=Path, default=None)
    parser.add_argument("--classical-results", type=Path, default=Path("results/baselines_classical_full_test.jsonl"))
    parser.add_argument("--examples-per-bucket", type=int, default=3)
    args = parser.parse_args()
    device = resolve_device(args.device)

    if args.epoch is not None:
        paths = resolve_matched_checkpoints(
            args.epoch, noref_checkpoint=args.noref_checkpoint,
            ref_checkpoint=args.ref_checkpoint,
            longer_noref_checkpoint=args.longer_noref_checkpoint,
        )
        args.noref_checkpoint = paths["noref"]
        args.ref_checkpoint = paths["reference"]
        args.longer_noref_checkpoint = paths["longer_noref"]
        if args.limit_per_bucket is None:
            args.limit_per_bucket = 50
        if args.output is None:
            args.output = Path(f"results/matched_epoch_{args.epoch}_eval.jsonl")
        if args.reconstruction_dir is None:
            args.reconstruction_dir = Path(f"results/matched_epoch_{args.epoch}_reconstructions")
    else:
        if args.noref_checkpoint is None:
            parser.error("--noref-checkpoint is required unless --epoch is supplied")
        if args.limit_per_bucket is None:
            args.limit_per_bucket = 100
        if args.output is None:
            args.output = Path("results/pipeline_eval.jsonl")
        if args.reconstruction_dir is None:
            args.reconstruction_dir = Path("results/reconstructions")

    noref = restore(get_baseline_codec(quality=3).to(device), args.noref_checkpoint, device,
                    expected_epoch=args.epoch)
    ref_codec = None
    if args.ref_checkpoint is not None and args.ref_checkpoint.exists():
        ref_codec = restore(ReferenceConditionedCodec(get_baseline_codec(quality=3)).to(device), args.ref_checkpoint, device,
                            expected_epoch=args.epoch)
    longer_noref = None
    if args.longer_noref_checkpoint is not None:
        longer_noref = restore(get_baseline_codec(quality=3).to(device), args.longer_noref_checkpoint, device,
                               expected_epoch=27)

    classical = {}
    if args.classical_results.exists():
        with args.classical_results.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                classical[record["pair_id"]] = record

    pairs = load_pairs(args.dataset, args.limit_per_bucket, args.limit_total, args.seed)
    if args.epoch is not None:
        missing_classical = [pair["pair_id"] for pair in pairs if pair["pair_id"] not in classical]
        wrong_bucket = [pair["pair_id"] for pair in pairs
                        if pair["pair_id"] in classical
                        and classical[pair["pair_id"]].get("degree_of_change") != pair["degree_of_change"]]
        if missing_classical or wrong_bucket:
            raise ValueError(f"classical baselines do not align with matched test pairs; missing={missing_classical[:5]}, bucket_mismatch={wrong_bucket[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.reconstruction_dir.mkdir(parents=True, exist_ok=True)
    example_counts = {}
    with args.output.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for pair in pairs:
            ref_tensor, _ = tensor_from_image(pair["pair_dir"] / "reference.png", device)
            target, _ = tensor_from_image(pair["pair_dir"] / "target.png", device)
            noref_result, noref_recon = score(noref(target), target)
            result = {"pair_id": pair["pair_id"], "degree_of_change": pair["degree_of_change"],
                      "lambda": args.lambda_value, "image_pixels": target.shape[-2] * target.shape[-1],
                      "noref": noref_result}
            if args.epoch is not None:
                result["matched_epoch"] = {"epoch_index": args.epoch, "epochs_completed": args.epoch + 1}
            if longer_noref is not None:
                result["noref_longer_trained_not_matched"] = score(longer_noref(target), target)[0]
                result["noref_longer_trained_label"] = "No-Reference epoch 27: longer-trained, not matched"
            noref_recon_path = None
            if example_counts.get(pair["degree_of_change"], 0) < args.examples_per_bucket:
                noref_recon_path = args.reconstruction_dir / f"{pair['pair_id']}_noref.png"
                save_tensor(noref_recon, noref_recon_path)
                result["noref_reconstruction"] = str(noref_recon_path)
            if ref_codec is not None:
                ref_result, ref_recon = score(ref_codec(ref_tensor, target), target)
                result["reference_conditioned"] = ref_result
                if noref_recon_path is not None:
                    ref_recon_path = args.reconstruction_dir / f"{pair['pair_id']}_reference.png"
                    save_tensor(ref_recon, ref_recon_path)
                    result["reference_reconstruction"] = str(ref_recon_path)
            result["classical"] = classical.get(pair["pair_id"])
            stream.write(json.dumps(clean_json(result), allow_nan=False) + "\n")
            if noref_recon_path is not None:
                example_counts[pair["degree_of_change"]] = example_counts.get(pair["degree_of_change"], 0) + 1
    print(f"Wrote {len(pairs)} pair records to {args.output}; device={device}; ref_codec={'available' if ref_codec else 'missing'}")


def save_tensor(tensor, path):
    import numpy as np
    array = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((array.clip(0, 1) * 255).round().astype("uint8"), "RGB").save(path)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()

"""Train the Phase 4 v1 reference-conditioned codec in FP32."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from compressai.losses import RateDistortionLoss
except ImportError:
    print("compressai not installed. Run this entrypoint in the configured training environment.", file=sys.stderr)
    raise

from data.dataset import ScreenshotDataset
from models.baseline_codec import get_baseline_codec
from models.conditional_codec import ReferenceConditionedCodec
from scripts.runtime import capture_rng_state, resolve_device, restore_rng_state, seed_everything, upload_epoch_artifacts


def train_epoch(model, loader, criterion, optimizer, device, epoch, max_steps, log_every):
    model.train()
    samples = []
    started = time.perf_counter()
    for step, (reference, target) in enumerate(loader):
        step_started = time.perf_counter()
        reference, target = reference.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(reference, target)
        losses = criterion(output, target)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0:
            sample = {
                "epoch": epoch, "step": step, "steps_per_epoch": len(loader),
                "loss": losses["loss"].item(), "mse": losses["mse_loss"].item(),
                "bpp": losses["bpp_loss"].item(),
                "elapsed_seconds": time.perf_counter() - started,
                "step_seconds": time.perf_counter() - step_started,
            }
            samples.append(sample)
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | loss={sample['loss']:.5f} "
                  f"mse={sample['mse']:.6f} bpp={sample['bpp']:.5f} "
                  f"step_s={sample['step_seconds']:.4f} elapsed_s={sample['elapsed_seconds']:.2f}", flush=True)
        if max_steps is not None and step + 1 >= max_steps:
            break
    samples.append({
        "epoch": epoch, "step": "epoch_complete",
        "steps_completed": min(len(loader), max_steps) if max_steps is not None else len(loader),
        "elapsed_seconds": time.perf_counter() - started,
    })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train reference-conditioned v1")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_pairs_full"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lmbda", type=float, default=0.0130)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/reference_v1"))
    parser.add_argument("--max-pairs-per-bucket", type=int, default=None)
    parser.add_argument("--metrics-output", type=Path, default=Path("results/reference_v1_metrics.json"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24041)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.device = str(resolve_device(args.device))
    seed_everything(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    dataset = ScreenshotDataset(args.dataset, split="train", patch_size=args.patch_size,
                                no_reference=False, max_pairs_per_bucket=args.max_pairs_per_bucket)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    model = ReferenceConditionedCodec(get_baseline_codec(quality=3)).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric="mse")

    start_epoch, history, resume_rng_state = 0, [], None
    if args.resume:
        checkpoints = sorted(args.save_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if checkpoints:
            checkpoint = torch.load(checkpoints[-1], map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            resume_rng_state = checkpoint.get("rng_state")
            print(f"Resuming from {checkpoints[-1]} at epoch {start_epoch}", flush=True)
        if args.metrics_output.exists():
            try:
                history = json.loads(args.metrics_output.read_text(encoding="utf-8")).get("history", [])
            except (json.JSONDecodeError, OSError):
                print("Previous metrics could not be read; starting new metric history", flush=True)

    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state, generator)
    else:
        seed_everything(args.seed)
        generator.manual_seed(args.seed)

    print(f"Training pairs: {len(dataset)} | device={args.device} | AMP=off (FP32)", flush=True)
    for epoch in range(start_epoch, args.epochs):
        history.extend(train_epoch(model, loader, criterion, optimizer, args.device,
                                   epoch, args.max_steps, args.log_every))
        model.update(force=True)
        checkpoint_path = args.save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {"lambda": args.lmbda, "patch_size": args.patch_size,
                       "batch_size": args.batch_size, "seed": args.seed,
                       "train_pairs": len(dataset)},
            "rng_state": capture_rng_state(generator),
        }, checkpoint_path)
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps({
            "dataset": str(args.dataset), "train_pairs": len(dataset),
            "max_pairs_per_bucket": args.max_pairs_per_bucket,
            "patch_size": args.patch_size, "batch_size": args.batch_size,
            "epochs": args.epochs, "lmbda": args.lmbda, "max_steps": args.max_steps,
            "seed": args.seed, "device": args.device, "amp": False, "history": history,
            "log_every": args.log_every,
        }, indent=2), encoding="utf-8")
        upload_epoch_artifacts(args.save_dir.name, args.save_dir, args.metrics_output, epoch)
        print(f"Saved {checkpoint_path}", flush=True)
        if Path("results/STOP_AFTER_EPOCH").exists():
            print("Pause requested; completed epoch checkpoint is saved. Resume with --resume after removing the stop request.", flush=True)
            raise SystemExit(75)


if __name__ == "__main__":
    main()

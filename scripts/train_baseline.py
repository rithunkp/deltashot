"""Train the No-Reference baseline codec."""
import argparse
import json
import time
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from compressai.losses import RateDistortionLoss
except ImportError:
    print("compressai not installed. Run scripts under WSL2 venv.", file=sys.stderr)
    sys.exit(1)

from models.baseline_codec import get_baseline_codec
from data.dataset import ScreenshotDataset
from scripts.runtime import capture_rng_state, resolve_device, restore_rng_state, seed_everything, upload_epoch_artifacts

def train_one_epoch(model, criterion, optimizer, train_dataloader, device, epoch, clip_max_norm, max_steps=None, log_every=10):
    model.train()
    history = []
    started = time.perf_counter()
    for i, d in enumerate(train_dataloader):
        step_started = time.perf_counter()
        d = d.to(device)
        optimizer.zero_grad()
        
        out_net = model(d)
        
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        if i % log_every == 0:
            metrics = {
                "epoch": epoch,
                "step": i,
                "steps_per_epoch": len(train_dataloader),
                "loss": out_criterion["loss"].item(),
                "mse": out_criterion["mse_loss"].item(),
                "bpp": out_criterion["bpp_loss"].item(),
                "elapsed_seconds": time.perf_counter() - started,
                "step_seconds": time.perf_counter() - step_started,
            }
            history.append(metrics)
            print(f"Epoch {epoch} | Step {i}/{len(train_dataloader)} | Loss: {metrics['loss']:.4f} | "
                  f"MSE: {metrics['mse']:.5f} | Bpp: {metrics['bpp']:.4f} | "
                  f"step_s: {metrics['step_seconds']:.4f} | elapsed_s: {metrics['elapsed_seconds']:.2f}", flush=True)
        if max_steps is not None and i + 1 >= max_steps:
            break
    history.append({
        "epoch": epoch,
        "step": "epoch_complete",
        "steps_completed": min(len(train_dataloader), max_steps) if max_steps is not None else len(train_dataloader),
        "elapsed_seconds": time.perf_counter() - started,
    })
    return history


def write_metrics(path, args, train_pairs, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset": str(args.dataset),
        "train_pairs": train_pairs,
        "max_pairs_per_bucket": args.max_pairs_per_bucket,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lmbda": args.lmbda,
        "max_steps": args.max_steps,
        "log_every": args.log_every,
        "seed": args.seed,
        "device": args.device,
        "amp": False,
        "history": history,
    }, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Train No-Reference Codec")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_pairs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lmbda", type=float, default=0.0130)
    parser.add_argument("--clip-max-norm", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/baseline"))
    parser.add_argument("--max-pairs-per-bucket", type=int, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest epoch checkpoint in --save-dir")
    parser.add_argument("--log-every", type=int, default=10, help="Record and print one timing sample every N steps")
    parser.add_argument("--seed", type=int, default=24041)
    args = parser.parse_args()

    args.device = str(resolve_device(args.device))
    seed_everything(args.seed)
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = ScreenshotDataset(
        args.dataset,
        split="train",
        patch_size=args.patch_size,
        no_reference=True,
        max_pairs_per_bucket=args.max_pairs_per_bucket,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=loader_generator)
    
    model = get_baseline_codec(quality=3)
    model = model.to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric="mse")

    start_epoch = 0
    resume_rng_state = None
    if args.resume:
        checkpoints = sorted(args.save_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if checkpoints:
            checkpoint = torch.load(checkpoints[-1], map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            resume_rng_state = checkpoint.get("rng_state")
            print(f"Resuming from {checkpoints[-1]} at epoch {start_epoch}", flush=True)
        else:
            print("--resume requested but no checkpoint exists; starting from epoch 0", flush=True)

    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state, loader_generator)
    else:
        seed_everything(args.seed)
        loader_generator.manual_seed(args.seed)
    
    print(f"Training pairs: {len(train_dataset)}", flush=True)
    history = []
    if args.resume and args.metrics_output is not None and args.metrics_output.exists():
        try:
            history = json.loads(args.metrics_output.read_text(encoding="utf-8")).get("history", [])
        except (json.JSONDecodeError, OSError):
            print(f"Could not read prior metrics from {args.metrics_output}; starting a fresh history", flush=True)
    for epoch in range(start_epoch, args.epochs):
        print(f"--- Epoch {epoch} ---")
        history.extend(train_one_epoch(model, criterion, optimizer, train_dataloader, args.device, epoch, args.clip_max_norm, args.max_steps, args.log_every))

        # update() freezes the quantized CDFs needed for actual compression at eval time.
        model.update(force=True)

        checkpoint_path = args.save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {"lambda": args.lmbda, "patch_size": args.patch_size, "batch_size": args.batch_size,
                       "seed": args.seed, "train_pairs": len(train_dataset)},
            "rng_state": capture_rng_state(loader_generator),
        }, checkpoint_path)
        print(f"Saved {checkpoint_path}", flush=True)
        if args.metrics_output is not None:
            write_metrics(args.metrics_output, args, len(train_dataset), history)
        upload_epoch_artifacts(args.save_dir.name, args.save_dir, args.metrics_output, epoch)
        if Path("results/STOP_AFTER_EPOCH").exists():
            print("Pause requested; completed epoch checkpoint is saved. Resume with --resume after removing the stop request.", flush=True)
            raise SystemExit(75)

if __name__ == "__main__":
    main()

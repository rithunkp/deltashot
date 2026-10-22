"""Self-contained balanced screenshot training entry point for Colab/Kaggle."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop, CenterCrop, ToTensor
from compressai.losses import RateDistortionLoss
from compressai.models import ScaleHyperprior


class TargetDataset(Dataset):
    def __init__(self, root, split, patch_size, pairs_per_bucket=None):
        self.split = split
        self.patch_size = patch_size
        self.crop = RandomCrop(patch_size) if split == "train" else CenterCrop(patch_size)
        self.to_tensor = ToTensor()
        pair_dirs = sorted(path for path in (Path(root) / split).iterdir() if path.is_dir())
        if pairs_per_bucket is not None:
            selected, counts = [], {}
            for path in pair_dirs:
                meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
                bucket = meta["degree_of_change"]
                if counts.get(bucket, 0) < pairs_per_bucket:
                    selected.append(path)
                    counts[bucket] = counts.get(bucket, 0) + 1
            pair_dirs = selected
        self.pair_dirs = pair_dirs

    def __len__(self):
        return len(self.pair_dirs)

    def __getitem__(self, index):
        image = Image.open(self.pair_dirs[index] / "target.png").convert("RGB")
        return self.to_tensor(self.crop(image))


def save_metrics(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lambda-value", type=float, default=0.013)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pairs-per-bucket", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/gpu_run"))
    parser.add_argument("--metrics", type=Path, default=Path("results/gpu_run_metrics.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = TargetDataset(args.dataset, "train", args.patch_size, args.pairs_per_bucket)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    model = ScaleHyperprior(N=128, M=192).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = RateDistortionLoss(lmbda=args.lambda_value, metric="mse")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    first_epoch, history = 0, []
    checkpoints = sorted(args.save_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if args.resume and checkpoints:
        state = torch.load(checkpoints[-1], map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        first_epoch = state["epoch"] + 1
        if args.metrics.exists():
            history = json.loads(args.metrics.read_text(encoding="utf-8")).get("history", [])

    print(f"device={device} pairs={len(dataset)} resolution={args.patch_size} batch={args.batch_size}", flush=True)
    for epoch in range(first_epoch, args.epochs):
        model.train()
        started = time.perf_counter()
        loss_sum = 0.0
        for step, image in enumerate(loader):
            image = image.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            output = model(image)
            losses = criterion(output, image)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += losses["loss"].item()
            print(f"epoch={epoch} step={step + 1}/{len(loader)} loss={losses['loss'].item():.5f}", flush=True)
        model.update(force=True)
        duration = time.perf_counter() - started
        summary = {"epoch": epoch, "mean_loss": loss_sum / max(1, len(loader)), "duration_seconds": duration}
        history.append(summary)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                   args.save_dir / f"checkpoint_epoch_{epoch}.pth")
        save_metrics(args.metrics, {"dataset": str(args.dataset), "pairs": len(dataset), "patch_size": args.patch_size,
                                    "batch_size": args.batch_size, "lambda": args.lambda_value,
                                    "device": str(device), "history": history})
        print(f"epoch={epoch} complete duration_seconds={duration:.2f}", flush=True)


if __name__ == "__main__":
    main()

"""Write a reproducible, read-only DeltaShot environment and artifact audit."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import compressai
import torch


BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")
SPLITS = ("train", "val", "test")


def dataset_summary(root: Path) -> dict[str, object]:
    counts: Counter[tuple[str, str]] = Counter()
    total = 0
    incomplete = 0
    for meta_path in root.glob("*/*/meta.json"):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        total += 1
        counts[(metadata.get("split", "unknown"), metadata.get("degree_of_change", "unknown"))] += 1
        if any(not (meta_path.parent / name).is_file() for name in ("reference.png", "target.png")):
            incomplete += 1
    return {
        "root": str(root),
        "total_pairs": total,
        "incomplete_pairs": incomplete,
        "counts": {split: {bucket: counts[(split, bucket)] for bucket in BUCKETS} for split in SPLITS},
        "split_totals": {split: sum(counts[(split, bucket)] for bucket in BUCKETS) for split in SPLITS},
    }


def file_summary(path: Path) -> dict[str, object]:
    return {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cuda_available = torch.cuda.is_available()
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except FileNotFoundError:
        nvidia_smi = "unavailable"

    checkpoint_root = repo_root / "checkpoints"
    audit = {
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "compressai_version": compressai.__version__,
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
            "cuda_vram_bytes": torch.cuda.get_device_properties(0).total_memory if cuda_available else 0,
            "nvidia_smi": nvidia_smi,
        },
        "dataset": dataset_summary(repo_root / "data" / "synthetic_pairs_full"),
        "artifacts": {
            "classical_full": file_summary(repo_root / "results" / "baselines_classical_full.jsonl"),
            "compressai_full": file_summary(repo_root / "results" / "baselines_compressai_full.jsonl"),
            "classical_legacy": file_summary(repo_root / "results" / "baselines_classical.jsonl"),
            "compressai_legacy": file_summary(repo_root / "results" / "baselines_compressai.jsonl"),
            "checkpoint_files": sorted(str(path.relative_to(repo_root)) for path in checkpoint_root.glob("**/*.pth")),
            "mode_selection_source": (repo_root / "models" / "mode_selection.py").is_file(),
            "conditional_codec_source": (repo_root / "models" / "conditional_codec.py").is_file(),
            "runbook": (repo_root / "RUNBOOK.md").is_file(),
        },
    }
    output = repo_root / "results" / "project_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

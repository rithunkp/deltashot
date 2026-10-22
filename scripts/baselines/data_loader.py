"""Dataset loader for DeltaShot synthetic pairs.

Each pair is a directory containing:
  reference.png   — the reference image (already stored)
  target.png     — the target image to compress
  meta.json      — metadata (bucket label, mutation type, chain position)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class SyntheticPair:
    """One reference-target pair from the synthetic dataset."""
    pair_id: str
    split: str
    reference_path: Path
    target_path: Path
    degree_of_change: str
    mutation_types: list[str]
    chain_position: int
    changed_pixel_ratio: float


def load_pairs(
    root: Path,
    split: str | None = None,
    bucket_filter: str | None = None,
) -> list[SyntheticPair]:
    """Load all synthetic pairs under `root`, optionally filtered by split or bucket."""
    pairs: list[SyntheticPair] = []
    splits = [split] if split else ("train", "val", "test")
    for sp in splits:
        sp_dir = root / sp
        if not sp_dir.is_dir():
            continue
        for pair_dir in sp_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            ref = pair_dir / "reference.png"
            tgt = pair_dir / "target.png"
            meta_path = pair_dir / "meta.json"
            if not (ref.exists() and tgt.exists() and meta_path.exists()):
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if bucket_filter and meta.get("degree_of_change") != bucket_filter:
                continue
            pairs.append(
                SyntheticPair(
                    pair_id=pair_dir.name,
                    split=sp,
                    reference_path=ref,
                    target_path=tgt,
                    degree_of_change=meta.get("degree_of_change", "unknown"),
                    mutation_types=meta.get("mutation_types", []),
                    chain_position=meta.get("chain_position", 1),
                    changed_pixel_ratio=meta.get("changed_pixel_ratio", 0.0),
                )
            )
    return pairs


def load_image(path: Path) -> Image.Image:
    """Load an image as RGB."""
    return Image.open(path).convert("RGB")

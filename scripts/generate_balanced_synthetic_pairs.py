"""Generate a quota-balanced DeltaShot dataset using the existing mutation pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from collections import Counter
from pathlib import Path
from random import Random

from data.mutations import HtmlPage
from data.rendering import HtmlRenderer
from scripts.generate_synthetic_pairs import (
    NON_REPLACEMENT_OPERATORS,
    generate_pair,
    load_pages,
)


BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")
DEFAULT_QUOTAS = {
    "train": {bucket: 4000 for bucket in BUCKETS},
    "val": {bucket: 400 for bucket in BUCKETS},
    "test": {bucket: 400 for bucket in BUCKETS},
}


def smoke_quotas(total: int) -> dict[str, dict[str, int]]:
    if total % 50:
        raise ValueError("smoke-test total must be divisible by 50 for balanced 80/10/10 splits")
    per_bucket = total // len(BUCKETS)
    return {
        "train": {bucket: per_bucket * 8 // 10 for bucket in BUCKETS},
        "val": {bucket: per_bucket // 10 for bucket in BUCKETS},
        "test": {bucket: per_bucket // 10 for bucket in BUCKETS},
    }


def mutation_plan(bucket: str, rng: Random) -> tuple[str, ...]:
    if bucket == "identical":
        return ()
    if bucket == "unrelated":
        return ("mutate_replace",)
    if bucket == "minor":
        return (rng.choice(("mutate_color", "mutate_text")),)
    if bucket == "moderate":
        return tuple(rng.choice(NON_REPLACEMENT_OPERATORS) for _ in range(2))
    return tuple(rng.choice(NON_REPLACEMENT_OPERATORS) for _ in range(4))


def existing_counts(output: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for meta_path in output.glob("*/*/meta.json"):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        counts[(metadata["split"], metadata["degree_of_change"])] += 1
    return counts


def next_pair_index(output: Path) -> int:
    """Allocate a new pair ID without colliding with interrupted prior runs."""
    indices = []
    for pair_dir in output.glob("*/pair_*"):
        try:
            indices.append(int(pair_dir.name.removeprefix("pair_")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def promote_candidate(candidate_dir: Path, destination_dir: Path) -> None:
    """Copy a rendered candidate from staging into the dataset and verify it.

    WSL cannot reliably rename non-empty directories on the mounted Windows
    filesystem, so accepted pairs are copied before their temporary staging
    directory is removed.
    """
    if destination_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing pair: {destination_dir}")

    shutil.copytree(candidate_dir, destination_dir)
    required = ("reference.png", "target.png", "meta.json")
    missing = [name for name in required if not (destination_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"candidate copy is incomplete at {destination_dir}: {missing}")
    shutil.rmtree(candidate_dir)


def next_target(quotas: dict[str, dict[str, int]], counts: Counter[tuple[str, str]]) -> tuple[str, str] | None:
    for split in ("train", "val", "test"):
        for bucket in BUCKETS:
            if counts[(split, bucket)] < quotas[split][bucket]:
                return split, bucket
    return None


async def generate_dataset(args: argparse.Namespace) -> Counter[tuple[str, str]]:
    pages: list[HtmlPage] = load_pages(args.input)
    if len(pages) < 2:
        raise ValueError("at least two WebSight pages are required for mutate_replace")

    quotas = DEFAULT_QUOTAS if args.full else smoke_quotas(args.count)
    for split in quotas:
        (args.output / split).mkdir(parents=True, exist_ok=True)
    counts = existing_counts(args.output)
    rng = Random(args.seed)
    staging = args.output / ".staging"
    attempts = 0
    accepted = sum(counts.values())
    pair_index = next_pair_index(args.output)
    max_attempts = args.max_attempts or sum(sum(values.values()) for values in quotas.values()) * 20

    async with HtmlRenderer(width=args.width, height=args.height) as renderer:
        while (target := next_target(quotas, counts)) is not None:
            if attempts >= max_attempts:
                raise RuntimeError(f"quota generation stopped after {attempts} attempts; remaining target={target}")
            split, requested_bucket = target
            candidate_name = f"candidate_{attempts:08d}"
            metadata = await generate_pair(
                attempts,
                pages,
                staging,
                rng,
                renderer,
                args.width,
                args.height,
                split=split,
                pair_name=candidate_name,
                mutation_chain=mutation_plan(requested_bucket, rng),
            )
            candidate_dir = staging / split / candidate_name
            actual_bucket = metadata["degree_of_change"]
            if actual_bucket == requested_bucket and counts[(split, actual_bucket)] < quotas[split][actual_bucket]:
                pair_name = f"pair_{pair_index:06d}"
                promote_candidate(candidate_dir, args.output / split / pair_name)
                counts[(split, actual_bucket)] += 1
                accepted += 1
                pair_index += 1
                if accepted % 100 == 0:
                    print(json.dumps({"accepted": accepted, "latest_bucket": actual_bucket}, sort_keys=True), flush=True)
            else:
                shutil.rmtree(candidate_dir)
            attempts += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a quota-balanced WebSight mutation dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_pairs_full"))
    parser.add_argument("--full", action="store_true", help="Generate the 20k/2k/2k target specified in PROJECT_STATE.")
    parser.add_argument("--count", type=int, default=100, help="Smoke-test train total, divisible by five.")
    parser.add_argument("--seed", type=int, default=24042)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()

    counts = asyncio.run(generate_dataset(args))
    print(json.dumps({f"{split}/{bucket}": count for (split, bucket), count in sorted(counts.items())}, indent=2))


if __name__ == "__main__":
    main()

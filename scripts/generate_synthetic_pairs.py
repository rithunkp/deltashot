from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from random import Random
from typing import Any

from data.bucketing import compute_change_stats
from data.mutations import MUTATION_OPERATORS, HtmlPage, MutationResult, mutate_replace
from data.rendering import HtmlRenderer


DEFAULT_SPLITS = ("train", "val", "test")


def load_pages(path: Path) -> list[HtmlPage]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [HtmlPage(html=record["html"], page_id=record["page_id"], metadata={k: v for k, v in record.items() if k not in {"html", "page_id"}}) for record in records]


def split_for_index(index: int) -> str:
    if index % 10 == 8:
        return "val"
    if index % 10 == 9:
        return "test"
    return "train"


NON_REPLACEMENT_OPERATORS = tuple(name for name in MUTATION_OPERATORS if name != "mutate_replace")


def choose_chain(index: int, rng: Random) -> tuple[str, ...]:
    """Cycle through one-step, cumulative, and unrelated pairs deterministically."""
    cycle_position = index % 12
    if cycle_position == 11:
        return ("mutate_replace",)
    if cycle_position < 5:
        return (NON_REPLACEMENT_OPERATORS[cycle_position],)
    chain_length = 2 if cycle_position < 8 else 4
    return tuple(rng.choice(NON_REPLACEMENT_OPERATORS) for _ in range(chain_length))


def apply_operator(page: HtmlPage, operator_name: str, pages: list[HtmlPage], rng: Random) -> MutationResult:
    if operator_name == "mutate_replace":
        return mutate_replace(page, rng, replacement_pages=pages)
    return MUTATION_OPERATORS[operator_name](page, rng)


async def generate_pair(
    pair_index: int,
    pages: list[HtmlPage],
    output_root: Path,
    rng: Random,
    renderer: HtmlRenderer,
    width: int,
    height: int,
    split: str | None = None,
    pair_name: str | None = None,
    mutation_chain: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    split = split or split_for_index(pair_index)
    pair_dir = output_root / split / (pair_name or f"pair_{pair_index:06d}")
    pair_dir.mkdir(parents=True, exist_ok=True)

    reference = rng.choice(pages)
    mutation_chain = mutation_chain if mutation_chain is not None else choose_chain(pair_index, rng)
    target = reference
    mutation_details = []
    for operator_name in mutation_chain:
        result = apply_operator(target, operator_name, pages, rng)
        target = result.page
        mutation_details.append({"type": result.mutation_type, "details": result.details})

    reference_path = pair_dir / "reference.png"
    target_path = pair_dir / "target.png"
    await renderer.render(reference.html, reference_path)
    await renderer.render(target.html, target_path)

    explicit_unrelated = any(step["details"].get("explicit_unrelated", False) for step in mutation_details)
    stats = compute_change_stats(reference_path, target_path, explicit_unrelated=explicit_unrelated)
    metadata = {
        "pair_id": pair_dir.name,
        "split": split,
        "reference_page_id": reference.page_id,
        "target_page_id": target.page_id,
        "mutation_type": "identity" if not mutation_chain else mutation_chain[0] if len(mutation_chain) == 1 else "mutation_chain",
        "mutation_types": list(mutation_chain),
        "mutation_details": mutation_details,
        "chain_position": len(mutation_chain),
        "changed_pixel_ratio": stats.changed_pixel_ratio,
        "degree_of_change": stats.bucket,
        "viewport": {"width": width, "height": height},
        "source": reference.metadata.get("source", "local_seed_pages"),
    }
    (pair_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


async def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = Random(args.seed)
    pages = load_pages(args.input)
    records = []
    async with HtmlRenderer(width=args.width, height=args.height) as renderer:
        for pair_index in range(args.count):
            records.append(await generate_pair(pair_index, pages, args.output, rng, renderer, args.width, args.height))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DeltaShot reference/target screenshot pairs.")
    parser.add_argument("--input", type=Path, default=Path("data/raw_websight/seed_pages.json"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_pairs"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=24041)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    records = asyncio.run(generate(args))
    summary: dict[str, int] = {}
    for record in records:
        summary[record["degree_of_change"]] = summary.get(record["degree_of_change"], 0) + 1
    print(json.dumps({"generated": len(records), "bucket_counts": summary}, indent=2))


if __name__ == "__main__":
    main()

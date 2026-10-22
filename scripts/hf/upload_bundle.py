"""Upload runnable code and a balanced, budgeted training/evaluation subset privately."""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")


def ignored_data(_, names):
    return {name for name in names if name.startswith("synthetic_pairs") or name == "raw_websight" or name == "__pycache__"}


def copy_subset(source, dest, split_counts):
    for split, per_bucket in split_counts.items():
        out_split = dest / split
        out_split.mkdir(parents=True, exist_ok=True)
        counts = {bucket: 0 for bucket in BUCKETS}
        for pair_dir in sorted((source / split).iterdir()):
            if not pair_dir.is_dir() or not (pair_dir / "meta.json").exists():
                continue
            meta = json.loads((pair_dir / "meta.json").read_text(encoding="utf-8"))
            bucket = meta.get("degree_of_change")
            if bucket not in counts or counts[bucket] >= per_bucket:
                continue
            shutil.copytree(pair_dir, out_split / pair_dir.name)
            counts[bucket] += 1
            if all(counts[name] >= per_bucket for name in BUCKETS):
                break
        if any(count != per_bucket for count in counts.values()):
            raise RuntimeError(f"Requested {per_bucket}/bucket for {split}; found {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="Private Hugging Face dataset repo, e.g. username/deltashot-bundle")
    parser.add_argument("--train-per-bucket", type=int, default=2000)
    parser.add_argument("--val-per-bucket", type=int, default=100)
    parser.add_argument("--test-per-bucket", type=int, default=100)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in the environment; the token is never read from an argument or printed.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deltashot-hf-bundle-") as temporary:
        payload = Path(temporary)
        shutil.copytree(ROOT / "scripts", payload / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(ROOT / "models", payload / "models", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(ROOT / "data", payload / "data", ignore=ignored_data)
        shutil.copytree(ROOT / "tests", payload / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "pyproject.toml", payload / "pyproject.toml")
        results = payload / "results"
        results.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "results/baselines_classical_full_test.jsonl", results / "baselines_classical_full_test.jsonl")
        copy_subset(
            ROOT / "data/synthetic_pairs_full", payload / "data/synthetic_pairs_full",
            {"train": args.train_per_bucket, "val": args.val_per_bucket, "test": args.test_per_bucket},
        )
        manifest = {
            "train_pairs_per_bucket": args.train_per_bucket,
            "val_pairs_per_bucket": args.val_per_bucket,
            "test_pairs_per_bucket": args.test_per_bucket,
            "train_pairs_total": args.train_per_bucket * 5,
            "test_pairs_total": args.test_per_bucket * 5,
            "training_steps_per_epoch_batch4": (args.train_per_bucket * 5 + 3) // 4,
            "subset_rationale": "10,000 training pairs (2,000/bucket) yields 2,500 optimizer steps/epoch at batch 4; this is a transfer- and compute-budgeted first cloud run. Actual T4 epoch timing is unmeasured and must be taken from calibrate-only before interpreting duration.",
        }
        (payload / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        api.upload_folder(
            repo_id=args.repo_id, repo_type="dataset", folder_path=str(payload),
            commit_message="Upload DeltaShot runnable pipeline and balanced training subset",
        )
    print(f"Uploaded private bundle {args.repo_id}: train={args.train_per_bucket * 5}, val={args.val_per_bucket * 5}, test={args.test_per_bucket * 5} pairs")


if __name__ == "__main__":
    main()

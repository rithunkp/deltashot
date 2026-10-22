"""Capture the demo pairs and precompute every available codec output."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from scripts.build_website_screenshot_pairs import build_manifest


ROOT = Path(__file__).resolve().parents[1]
GALLERY_MANIFEST = ROOT / "website_screenshots" / "manifest.json"
OUTPUT = ROOT / "results" / "premade_demos"
RUNS = OUTPUT / "runs"


def _compact_result(item):
    return {key: value for key, value in item.items() if key != "image_data"}


def main():
    existing = json.loads(GALLERY_MANIFEST.read_text(encoding="utf-8")) if GALLERY_MANIFEST.is_file() else {}
    real_pairs = [pair for pair in existing.get("pairs", [])
                  if pair.get("origin") == "real" and pair.get("id", "").startswith(("python_", "bootstrap_"))]
    if len(real_pairs) < 5:
        build_manifest(capture=True)
    gallery = json.loads(GALLERY_MANIFEST.read_text(encoding="utf-8"))
    captured_real = [pair for pair in gallery.get("pairs", [])
                     if pair.get("origin") == "real" and pair.get("id", "").startswith(("python_", "bootstrap_"))]
    if len(captured_real) < 5:
        raise RuntimeError(
            f"Only {len(captured_real)} real version pairs passed capture checks; at least five are required. "
            "Review website_screenshots/manifest.json dropped reasons and rerun after fixing the source/capture issue."
        )

    from scripts import local_demo
    local_demo.OUTPUT_ROOT = RUNS
    RUNS.mkdir(parents=True, exist_ok=True)
    runner = local_demo.CodecRunner()
    records = []
    for index, pair in enumerate(gallery["pairs"], 1):
        reference_path = ROOT / "website_screenshots" / pair["reference"]
        target_path = ROOT / "website_screenshots" / pair["target"]
        if not reference_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"pair {pair['id']} is missing an image: {reference_path} / {target_path}")
        with Image.open(reference_path) as ref_file, Image.open(target_path) as target_file:
            reference, target = ref_file.convert("RGB"), target_file.convert("RGB")
        response = runner.run(target, target_path.stat().st_size, reference)
        run_id = response["run_id"]
        records.append({
            **pair,
            "run_id": run_id,
            "working_size": response["manifest"]["working_size"],
            "mode_selection": response["manifest"]["mode_selection"],
            "neural_decode_source": response["manifest"]["neural_decode_source"],
            "methods": [_compact_result(item) for item in response["results"]],
        })
        print(f"[{index}/{len(gallery['pairs'])}] encoded {pair['id']} ({pair['origin']}, {pair.get('bucket')})", flush=True)

    output_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": "website_screenshots/manifest.json",
        "codec_mode": "CompressAI checkpoints at epoch 27; classical codecs computed on the same 512x512 working image",
        "training_results_are_not_benchmarks": True,
        "pairs": records,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "manifest.json"
    destination.write_text(json.dumps(output_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {len(records)} premade comparisons to {destination}")


if __name__ == "__main__":
    main()

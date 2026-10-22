"""Post-hoc per-pair mode choice with a one-bit cost for reference mode."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.matched_epoch import resolve_matched_checkpoints


BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")


def mean_finite(values):
    finite = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return statistics.mean(finite) if finite else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/pipeline_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/pipeline_eval_modes.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/pipeline_summary.json"))
    parser.add_argument("--epoch", type=int, help="Zero-based matched checkpoint epoch")
    parser.add_argument("--noref-checkpoint", type=Path)
    parser.add_argument("--ref-checkpoint", type=Path)
    parser.add_argument("--longer-noref-checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint_paths = None
    if args.epoch is not None:
        checkpoint_paths = resolve_matched_checkpoints(
            args.epoch, noref_checkpoint=args.noref_checkpoint,
            ref_checkpoint=args.ref_checkpoint,
            longer_noref_checkpoint=args.longer_noref_checkpoint,
        )
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_bucket = defaultdict(list)
    any_reference = any(record.get("reference_conditioned") for record in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            bucket = record.get("degree_of_change", "unknown")
            noref = record["noref"]
            selected = {**noref, "codec": "no_reference", "flag_bits": 0}
            mode_record = {"status": "UNAVAILABLE", "reason": "reference checkpoint/result missing"}
            if record.get("reference_conditioned"):
                ref = record["reference_conditioned"]
                pixels = record["image_pixels"]
                flag_bpp = 1.0 / pixels
                lambda_value = record["lambda"]
                j_noref = noref["estimated_bpp_from_likelihoods"] + lambda_value * noref["mse"]
                j_ref_with_flag = ref["estimated_bpp_from_likelihoods"] + flag_bpp + lambda_value * ref["mse"]
                use_reference = j_ref_with_flag < j_noref
                if use_reference:
                    selected = {**ref, "codec": "reference_conditioned", "flag_bits": 1,
                                "estimated_bpp_from_likelihoods": ref["estimated_bpp_from_likelihoods"] + flag_bpp}
                j_mode = selected["estimated_bpp_from_likelihoods"] + lambda_value * selected["mse"]
                mode_record = {
                    "codec": selected["codec"], "flag_bits": selected["flag_bits"],
                    "estimated_bpp_from_likelihoods": selected["estimated_bpp_from_likelihoods"],
                    "psnr_db": selected["psnr_db"], "mse": selected["mse"], "ms_ssim": selected["ms_ssim"],
                    "objective_j": j_mode, "noref_objective_j": j_noref,
                }
            record["mode_selection"] = mode_record
            by_bucket[bucket].append(record)
            output.write(json.dumps(record, allow_nan=False) + "\n")

    summaries = {}
    for bucket in BUCKETS:
        group = by_bucket.get(bucket, [])
        summaries[bucket] = {
            "pair_count": len(group),
            "no_reference": summarize([row.get("noref") for row in group], summary_lambda(group)),
            "reference_conditioned": summarize([row.get("reference_conditioned") for row in group], summary_lambda(group)),
            "longer_trained_no_reference_not_matched": summarize(
                [row.get("noref_longer_trained_not_matched") for row in group], summary_lambda(group)),
            "mode_selection": summarize([row.get("mode_selection") for row in group]),
            "classical": summarize_classical([row.get("classical") for row in group]),
        }

    unrelated = by_bucket.get("unrelated", [])
    if any_reference and unrelated:
        pass_bar = all(row["mode_selection"]["objective_j"] <= row["mode_selection"]["noref_objective_j"] + 1e-12
                       for row in unrelated)
        mode_status = "PASS" if pass_bar else "FAIL"
        mode_reason = ("Every unrelated pair's selected J (including a 1-bit flag when reference mode is selected) "
                       "is <= its No-Reference J" if pass_bar else "At least one unrelated pair has selected J above No-Reference J")
    else:
        mode_status = "FAIL"
        mode_reason = "Reference-conditioned results or unrelated-bucket samples are missing; correctness bar not evaluated"
    summary = {
        "lambda": records[0].get("lambda") if records else None,
        "epoch": args.epoch,
        "epochs_completed": args.epoch + 1 if args.epoch is not None else None,
        "checkpoint_paths": ({key: str(value) if value is not None else None
                              for key, value in checkpoint_paths.items()} if checkpoint_paths else None),
        "mode_selection": {
            "status": mode_status, "unrelated_correctness": mode_status, "reason": mode_reason,
            "rule": "J_noref=bpp_noref+lambda*MSE_noref; J_ref=bpp_ref+1/(H*W)+lambda*MSE_ref. Choose the lower J; no-reference is the implicit default (0 flag bits), and reference mode adds a 1-bit flag to bpp.",
            "objective": "bpp + lambda * raw MSE on [0,1] pixels; likelihood-derived bpp is an estimate, not a written bitstream size",
        },
        "by_bucket": summaries,
        "reduced_scale": True,
        "source": str(args.input),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Mode selection {mode_status}: {mode_reason}")


def summary_lambda(rows):
    return rows[0].get("lambda") if rows else 0.0


def summarize(rows, lambda_value=0.0):
    rows = [row for row in rows if isinstance(row, dict) and "estimated_bpp_from_likelihoods" in row]
    return {
        "estimated_bpp_from_likelihoods": mean_finite([r["estimated_bpp_from_likelihoods"] for r in rows]),
        "psnr_db": mean_finite([r.get("psnr_db") for r in rows]),
        "ms_ssim": mean_finite([r.get("ms_ssim") for r in rows]),
        "mse": mean_finite([r.get("mse") for r in rows]),
        "objective_j": mean_finite([r["estimated_bpp_from_likelihoods"] + lambda_value * r["mse"] for r in rows]),
    }


def summarize_classical(rows):
    rows = [row for row in rows if isinstance(row, dict)]
    result = {}
    for codec in ("jpeg", "webp"):
        points = [point for row in rows for point in row.get(codec, []) if point.get("quality") == 75]
        result[f"{codec}_q75"] = {
            "bpp": mean_finite([point.get("bits_per_pixel") for point in points]),
            "psnr_db": mean_finite([point.get("psnr_db") for point in points]),
        }
    delta = [row.get("delta_png") for row in rows if row.get("delta_png")]
    result["delta_png"] = {
        "bpp": mean_finite([point.get("bits_per_pixel") for point in delta]),
        "psnr_db": mean_finite([point.get("psnr_db") for point in delta]),
    }
    return result


if __name__ == "__main__":
    main()

"""Resumable DeltaShot CPU/GPU pipeline; no AMP is used with entropy models."""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.optim as optim
from compressai.losses import RateDistortionLoss
from torch.utils.data import DataLoader

from data.dataset import ScreenshotDataset
from models.baseline_codec import get_baseline_codec
from models.conditional_codec import ReferenceConditionedCodec
from scripts.matched_epoch import resolve_matched_checkpoints
from scripts.runtime import resolve_device, seed_everything


ROOT = Path.cwd()
RESULTS = ROOT / "results"
STATUS = RESULTS / "pipeline_status.txt"
BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")
ALL_STAGES = ("preflight", "calibrate", "train_noref", "train_ref", "evaluate", "mode_selection", "report")


def append_status(stage, result, reason):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with STATUS.open("a", encoding="utf-8") as stream:
        stream.write(f"{stamp} | {stage} | {result} | {reason}\n")


def last_statuses():
    statuses = {}
    if STATUS.exists():
        for line in STATUS.read_text(encoding="utf-8").splitlines():
            fields = line.split(" | ", 3)
            if len(fields) == 4:
                statuses[fields[1]] = fields[2]
    return statuses


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def model_for(kind):
    base = get_baseline_codec(quality=3)
    return base if kind == "noref" else ReferenceConditionedCodec(base)


def make_loader(dataset, batch_size, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def preflight(device, seed, skip_pytest=False):
    test_log = RESULTS / "pipeline_pytest.txt"
    if skip_pytest:
        pytest_status = "SKIPPED"
    else:
        code = run_logged([sys.executable, "-m", "pytest", "-q"], test_log)
        if code:
            raise RuntimeError(f"pytest failed (exit {code}); full output: {test_log}")
        pytest_status = "PASS"

    smoke_root = RESULTS / "preflight"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke = {"pytest": pytest_status, "noref": "FAIL", "reference_conditioned": "FAIL", "eval": "FAIL"}
    shared = ["--dataset", "data/synthetic_pairs_full", "--epochs", "1", "--batch-size", "4",
              "--patch-size", "128", "--lmbda", "0.0130", "--device", str(device),
              "--max-pairs-per-bucket", "4", "--max-steps", "5", "--log-every", "1", "--seed", str(seed)]
    noref_dir = smoke_root / "noref"
    noref_cmd = [sys.executable, "-m", "scripts.train_baseline", *shared,
                 "--save-dir", str(noref_dir), "--metrics-output", str(smoke_root / "noref_metrics.json")]
    if run_logged(noref_cmd, smoke_root / "noref.log"):
        raise RuntimeError(f"No-Reference 5-step smoke training failed; see {smoke_root / 'noref.log'}")
    smoke["noref"] = "PASS"

    ref_dir = smoke_root / "reference"
    ref_cmd = [sys.executable, "-m", "scripts.train_reference", *shared,
               "--save-dir", str(ref_dir), "--metrics-output", str(smoke_root / "reference_metrics.json")]
    reference_ok = run_logged(ref_cmd, smoke_root / "reference.log") == 0
    smoke["reference_conditioned"] = "PASS" if reference_ok else "FAIL"
    reference_smoke_failed = not reference_ok

    eval_cmd = [sys.executable, "-m", "scripts.evaluate_codecs", "--dataset", "data/synthetic_pairs_full",
                "--noref-checkpoint", str(noref_dir / "checkpoint_epoch_0.pth"), "--device", str(device),
                "--limit-total", "8", "--output", str(smoke_root / "eval.jsonl"),
                "--reconstruction-dir", str(smoke_root / "reconstructions")]
    (smoke_root / "eval.jsonl").unlink(missing_ok=True)
    if reference_ok:
        eval_cmd.extend(["--ref-checkpoint", str(ref_dir / "checkpoint_epoch_0.pth")])
    combined_eval_failed = False
    if run_logged(eval_cmd, smoke_root / "eval.log"):
        if not reference_ok:
            raise RuntimeError(f"No-Reference 8-pair evaluation failed; full output: {smoke_root / 'eval.log'}")
        combined_eval_failed = True
        reference_ok = False
        smoke["reference_conditioned"] = "FAIL"
        eval_cmd = [arg for arg in eval_cmd if arg not in ("--ref-checkpoint", str(ref_dir / "checkpoint_epoch_0.pth"))]
        if run_logged(eval_cmd, smoke_root / "eval_noref_only.log"):
            raise RuntimeError(f"No-Reference fallback evaluation failed; full output: {smoke_root / 'eval_noref_only.log'}")
        smoke["eval"] = "PARTIAL"
    else:
        smoke["eval"] = "PASS" if reference_ok else "PARTIAL"
    (RESULTS / "preflight.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    reason = ", ".join(f"{name}={result}" for name, result in smoke.items())
    if reference_smoke_failed:
        reason += f"; reference smoke full output/traceback: {smoke_root / 'reference.log'}"
    if smoke["eval"] != "PASS":
        eval_log = "eval.log" if combined_eval_failed else "eval_noref_only.log"
        reason += f"; evaluation output/traceback: {smoke_root / eval_log}"
    return smoke, reason


def benchmark(kind, crop_size, device, steps, seed, data_root):
    seed_everything(seed)
    dataset = ScreenshotDataset(data_root, split="train", patch_size=crop_size,
                                no_reference=(kind == "noref"), max_pairs_per_bucket=32)
    loader = make_loader(dataset, 4, seed)
    model = model_for(kind).to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = RateDistortionLoss(lmbda=0.0130, metric="mse")
    timings = []
    iterator = iter(loader)
    for index in range(steps + 2):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if kind == "noref":
            reference = None
            target = batch.to(device)
        else:
            reference, target = (part.to(device) for part in batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(target) if kind == "noref" else model(reference, target)
        loss = criterion(output, target)["loss"]
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= 2:
            timings.append(elapsed)
    return {"seconds_per_step": statistics.median(timings), "samples_seconds": timings}


def calibrate(device, args, reference_ok):
    sizes = {}
    kinds = ["noref", "reference"] if reference_ok else ["noref"]
    for crop in (128, 256):
        sizes[str(crop)] = {}
        for kind in kinds:
            sizes[str(crop)][kind] = benchmark(kind, crop, device, args.calibration_steps, args.seed, args.dataset)

    budget_seconds = args.budget_hours * 3600
    active_count = len(kinds)
    per_codec_seconds = budget_seconds * (0.4 if active_count == 2 else 0.8)
    slowest_256 = max(sizes["256"][kind]["seconds_per_step"] for kind in kinds)
    epochs_if_full_256 = per_codec_seconds / (math.ceil(20000 / args.batch_size) * slowest_256)
    crop = 256 if epochs_if_full_256 >= 8 else 128
    slowest = max(sizes[str(crop)][kind]["seconds_per_step"] for kind in kinds)
    target_steps_for_eight = int((per_codec_seconds * 0.85) / (8 * slowest))
    pairs = min(20000, max(20, (target_steps_for_eight * args.batch_size // 20) * 20))
    steps_per_epoch = math.ceil(pairs / args.batch_size)
    epochs = max(1, min(args.max_epochs, int((per_codec_seconds * 0.85) / (steps_per_epoch * slowest))))
    allocation = {"noref_seconds": per_codec_seconds, "reference_seconds": per_codec_seconds if reference_ok else 0,
                  "evaluation_seconds": budget_seconds * 0.15,
                  "fallback_noref_seconds": per_codec_seconds if not reference_ok else 0}
    plan = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "amp": False, "lambda": 0.0130, "seed": args.seed, "batch_size": args.batch_size,
        "calibration": sizes, "crop_size": crop, "pairs_per_epoch": pairs,
        "pairs_per_bucket": pairs // 5, "epochs": epochs, "estimated_step_seconds": slowest,
        "budget_hours": args.budget_hours, "allocations": allocation,
        "reduced_scale": crop != 256 or pairs < 20000 or epochs < 30,
        "policy": "128px default; 256px only when >=8 full-20k epochs fit in its training allocation",
    }
    (RESULTS / "pipeline_calibration.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def run_training(kind, plan, device, resume, data_root):
    name = "noref" if kind == "noref" else "reference"
    module = "scripts.train_baseline" if kind == "noref" else "scripts.train_reference"
    run_id = f"pipeline_{name}_lambda_00130"
    command = [sys.executable, "-m", module, "--dataset", str(data_root),
               "--epochs", str(plan["epochs"]), "--batch-size", str(plan["batch_size"]),
               "--patch-size", str(plan["crop_size"]), "--lmbda", "0.0130", "--device", str(device),
               "--save-dir", f"checkpoints/{run_id}", "--max-pairs-per-bucket", str(plan["pairs_per_bucket"]),
               "--metrics-output", f"results/{run_id}_metrics.json", "--seed", str(plan["seed"])]
    if resume:
        command.append("--resume")
    return run_logged(command, RESULTS / f"{run_id}.log"), run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_pairs_full"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--eval-pairs-per-bucket", type=int, default=None)
    parser.add_argument("--epoch", type=int, help="Evaluate a matched zero-based checkpoint epoch; epoch 9 means 10 epochs completed")
    parser.add_argument("--noref-checkpoint", type=Path)
    parser.add_argument("--ref-checkpoint", type=Path)
    parser.add_argument("--longer-noref-checkpoint", type=Path)
    parser.add_argument("--calibration-steps", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=24041)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES))
    parser.add_argument("--resume", action="store_true", help="Skip passed stages and resume from epoch checkpoints")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip only the pytest stage of preflight")
    args = parser.parse_args()
    stop_file = RESULTS / "STOP_AFTER_EPOCH"
    if args.resume and stop_file.exists():
        stop_file.unlink()
    device = resolve_device(args.device)
    stages = ["preflight", "calibrate"] if args.calibrate_only else args.stages
    if args.eval_pairs_per_bucket is None:
        args.eval_pairs_per_bucket = 50 if args.epoch is not None else 100
    matched_paths = None
    if args.epoch is not None:
        try:
            matched_paths = resolve_matched_checkpoints(
                args.epoch, noref_checkpoint=args.noref_checkpoint,
                ref_checkpoint=args.ref_checkpoint,
                longer_noref_checkpoint=args.longer_noref_checkpoint,
            )
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    if not args.resume and STATUS.exists():
        STATUS.write_text("", encoding="utf-8")
    results_status = last_statuses()
    preflight_data = None
    plan = None
    ref_ok = True

    if "preflight" in stages:
        if args.resume and results_status.get("preflight") == "PASS" and (RESULTS / "preflight.json").exists():
            preflight_data = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
            ref_ok = preflight_data.get("reference_conditioned") == "PASS"
        else:
            try:
                preflight_data, reason = preflight(device, args.seed, skip_pytest=args.skip_pytest)
                ref_ok = preflight_data.get("reference_conditioned") == "PASS"
                complete = (preflight_data.get("noref") == "PASS"
                            and preflight_data.get("reference_conditioned") == "PASS"
                            and preflight_data.get("eval") == "PASS")
                append_status("preflight", "PASS" if complete else "FAIL", reason)
            except Exception as error:
                failure_path = RESULTS / "preflight_failure.log"
                failure_path.write_text(traceback.format_exc(), encoding="utf-8")
                append_status("preflight", "FAIL", f"{error}; full traceback: {failure_path}")
                print(f"Preflight failed: {error}", file=sys.stderr)
                return 1

    if "calibrate" in stages:
        if args.resume and results_status.get("calibrate") == "PASS" and (RESULTS / "pipeline_calibration.json").exists():
            plan = json.loads((RESULTS / "pipeline_calibration.json").read_text(encoding="utf-8"))
        else:
            try:
                if preflight_data is None:
                    preflight_path = RESULTS / "preflight.json"
                    if not preflight_path.exists():
                        raise RuntimeError("preflight results missing; run preflight first")
                    preflight_data = json.loads(preflight_path.read_text(encoding="utf-8"))
                    ref_ok = preflight_data.get("reference_conditioned") == "PASS"
                plan = calibrate(device, args, ref_ok)
                reason = f"calibrated active codecs; crop={plan['crop_size']}, pairs/epoch={plan['pairs_per_epoch']}, epochs={plan['epochs']}, AMP=off"
                append_status("calibrate", "PASS", reason)
            except Exception as error:
                append_status("calibrate", "FAIL", str(error))
                print(f"Calibration failed: {error}", file=sys.stderr)
                return 1
    if args.calibrate_only:
        print("Calibration-only run complete; no model training was started.")
        return 0

    if any(stage in stages for stage in ("train_noref", "train_ref")) and plan is None:
        path = RESULTS / "pipeline_calibration.json"
        if not path.exists():
            print("Calibration plan missing; run the calibrate stage first.", file=sys.stderr)
            return 1
        plan = json.loads(path.read_text(encoding="utf-8"))
        preflight_path = RESULTS / "preflight.json"
        if preflight_path.exists():
            ref_ok = json.loads(preflight_path.read_text(encoding="utf-8")).get("reference_conditioned") == "PASS"

    for stage, kind in (("train_noref", "noref"), ("train_ref", "reference")):
        if stage not in stages:
            continue
        if kind == "reference" and not ref_ok:
            append_status(stage, "SKIP", "Reference-conditioned preflight failed; training skipped; its compute allocation was assigned to No-Reference during calibration")
            continue
        statuses = last_statuses()
        if args.resume and statuses.get(stage) == "PASS":
            continue
        code, run_id = run_training(kind, plan, device, args.resume, args.dataset)
        if code == 75:
            append_status(stage, "PAUSED", f"safe pause after a completed epoch checkpoint for {run_id}; rerun with --resume to continue")
            print(f"Pipeline paused at an epoch boundary; checkpoint saved. Resume with --resume. Status: {STATUS}")
            return 0
        if code:
            append_status(stage, "FAIL", f"training subprocess exited {code}; inspect results/{run_id}.log and resume with --resume")
            return 1
        append_status(stage, "PASS", f"lambda=0.0130 crop={plan['crop_size']} pairs/epoch={plan['pairs_per_epoch']} epochs={plan['epochs']} device={device} AMP=off")

    if "evaluate" in stages and not (args.resume and args.epoch is None and last_statuses().get("evaluate") == "PASS"):
        noref_dir = Path("checkpoints/pipeline_noref_lambda_00130")
        ref_dir = Path("checkpoints/pipeline_reference_lambda_00130")
        if matched_paths:
            noref_checkpoint = matched_paths["noref"]
            selected_ref_checkpoint = matched_paths["reference"]
            longer_noref_checkpoint = matched_paths["longer_noref"]
            eval_output = Path(f"results/matched_epoch_{args.epoch}_eval.jsonl")
            reconstruction_output = Path(f"results/matched_epoch_{args.epoch}_reconstructions")
        else:
            noref_checkpoint = args.noref_checkpoint
            selected_ref_checkpoint = args.ref_checkpoint
            longer_noref_checkpoint = args.longer_noref_checkpoint
            eval_output = Path("results/pipeline_eval.jsonl")
            reconstruction_output = Path("results/reconstructions")
            if noref_checkpoint is None:
                noref_checkpoint = noref_dir / f"checkpoint_epoch_{plan['epochs'] - 1}.pth" if plan else None
        if noref_checkpoint is None or not noref_checkpoint.exists():
            checkpoints = sorted(noref_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
            noref_checkpoint = checkpoints[-1] if checkpoints else None
        if noref_checkpoint is None:
            append_status("evaluate", "FAIL", "No trained No-Reference checkpoint found")
            return 1
        command = [sys.executable, "-m", "scripts.evaluate_codecs", "--dataset", str(args.dataset),
                   "--noref-checkpoint", str(noref_checkpoint), "--device", str(device),
                   "--lambda-value", "0.0130", "--limit-per-bucket", str(args.eval_pairs_per_bucket),
                   "--output", str(eval_output), "--reconstruction-dir", str(reconstruction_output)]
        if args.epoch is not None:
            command.extend(["--epoch", str(args.epoch)])
        if selected_ref_checkpoint is None:
            ref_checkpoints = sorted(ref_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
            selected_ref_checkpoint = ref_checkpoints[-1] if ref_checkpoints else None
        if selected_ref_checkpoint is not None and ref_ok:
            command.extend(["--ref-checkpoint", str(selected_ref_checkpoint)])
        if longer_noref_checkpoint is not None:
            command.extend(["--longer-noref-checkpoint", str(longer_noref_checkpoint)])
        code = run_logged(command, RESULTS / "pipeline_evaluate.log")
        if code:
            append_status("evaluate", "FAIL", f"evaluation subprocess exited {code}; see results/pipeline_evaluate.log")
            return 1
        epoch_note = f", matched epoch={args.epoch} ({args.epoch + 1} epochs completed)" if args.epoch is not None else ""
        append_status("evaluate", "PASS", f"test subset={args.eval_pairs_per_bucket}/bucket{epoch_note}, resolution=512, device={device}; likelihood bpp is estimated")

    if "mode_selection" in stages and not (args.resume and args.epoch is None and last_statuses().get("mode_selection") == "PASS"):
        ref_dir = Path("checkpoints/pipeline_reference_lambda_00130")
        ref_checkpoints = list(ref_dir.glob("checkpoint_epoch_*.pth"))
        if matched_paths:
            eval_input = Path(f"results/matched_epoch_{args.epoch}_eval.jsonl")
            mode_output = Path(f"results/matched_epoch_{args.epoch}_modes.jsonl")
            summary_output = Path(f"results/matched_epoch_{args.epoch}_summary.json")
            noref_checkpoint = matched_paths["noref"]
            selected_ref_checkpoint = matched_paths["reference"]
            longer_noref_checkpoint = matched_paths["longer_noref"]
        else:
            eval_input = Path("results/pipeline_eval.jsonl")
            mode_output = Path("results/pipeline_eval_modes.jsonl")
            summary_output = Path("results/pipeline_summary.json")
            noref_checkpoint = args.noref_checkpoint
            selected_ref_checkpoint = args.ref_checkpoint or (ref_checkpoints[-1] if ref_checkpoints else None)
            longer_noref_checkpoint = args.longer_noref_checkpoint
        if selected_ref_checkpoint is None or not Path(selected_ref_checkpoint).exists():
            append_status("mode_selection", "SKIP", "Reference-conditioned checkpoint unavailable; unrelated-bucket correctness was not evaluated")
        else:
            mode_command = [sys.executable, "-m", "scripts.select_modes", "--input", str(eval_input),
                            "--output", str(mode_output), "--summary", str(summary_output)]
            if args.epoch is not None:
                mode_command.extend(["--epoch", str(args.epoch), "--noref-checkpoint", str(noref_checkpoint),
                                     "--ref-checkpoint", str(selected_ref_checkpoint)])
                if longer_noref_checkpoint is not None:
                    mode_command.extend(["--longer-noref-checkpoint", str(longer_noref_checkpoint)])
            code = run_logged(mode_command,
                              RESULTS / "pipeline_mode_selection.log")
            summary_path = summary_output
            if code or not summary_path.exists():
                append_status("mode_selection", "FAIL", f"mode selection failed (exit {code}); see results/pipeline_mode_selection.log")
                return 1
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            status = "PASS" if summary.get("mode_selection", {}).get("unrelated_correctness") == "PASS" else "FAIL"
            reason = summary.get("mode_selection", {}).get("reason", "missing mode-selection evaluation")
            append_status("mode_selection", status, reason)

    if "report" in stages and not (args.resume and args.epoch is None and last_statuses().get("report") == "PASS"):
        report_command = [sys.executable, "-m", "scripts.build_pipeline_report"]
        if args.epoch is not None:
            report_command.extend(["--epoch", str(args.epoch),
                                   "--matched-eval", f"results/matched_epoch_{args.epoch}_modes.jsonl",
                                   "--matched-summary", f"results/matched_epoch_{args.epoch}_summary.json",
                                   "--noref-checkpoint", str(matched_paths["noref"]),
                                   "--ref-checkpoint", str(matched_paths["reference"])])
            if matched_paths["longer_noref"] is not None:
                report_command.extend(["--longer-noref-checkpoint", str(matched_paths["longer_noref"])])
        code = run_logged(report_command, RESULTS / "pipeline_report.log")
        if code:
            append_status("report", "FAIL", f"report generation exited {code}; see results/pipeline_report.log")
            return 1
        append_status("report", "PASS", "results/demo.html regenerated from pipeline artifacts")

    print(f"Pipeline stages complete or skipped. Status: {STATUS}; resolved device: {device}")
    final_statuses = last_statuses()
    return 1 if any(final_statuses.get(stage) == "FAIL" for stage in stages) else 0


if __name__ == "__main__":
    raise SystemExit(main())

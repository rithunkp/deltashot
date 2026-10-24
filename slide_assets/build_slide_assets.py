"""Build slide tables/charts from existing results/ artifacts only."""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent
BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")
FOOTNOTE = "500 pairs, 100 per bucket, one lambda (0.0130), neural bpp estimated from model likelihoods."
METHODS = ("JPEG", "WebP", "Delta+PNG", "CompressAI", "No-Ref", "Ref-Cond", "Mode Sel")
COLORS = ("#d55e00", "#007f86", "#7b7b7b", "#6f55a5", "#276fbf", "#17824e", "#bc4b68")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=5) -> str:
    if value is None:
        return "missing"
    if isinstance(value, str):
        return value
    if math.isnan(float(value)):
        return "missing"
    if math.isinf(float(value)):
        return "+inf"
    return f"{float(value):.{digits}f}"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    if any(math.isnan(float(value)) for value in values):
        raise ValueError("NaN found in source results")
    return math.fsum(values) / len(values)


def font(size: int, bold: bool = False):
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def save_table_png(path: Path, rows: list[dict], quality_rows: list[dict]) -> None:
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    title_f, sub_f, header_f, body_f, foot_f = font(38, True), font(20), font(19, True), font(21), font(16)
    draw.text((72, 46), "Results: Bits per Pixel by Degree of Change", fill="#142330", font=title_f)
    draw.text((74, 104), "Same 500 evaluation pairs; baseline rates use the nearest finite-PSNR setting to Ref-Cond.", fill="#445663", font=sub_f)
    x0, y0 = 55, 158
    widths = [180, 190, 190, 190, 190, 190, 190, 190]
    headers = ["Degree of change", "JPEG", "WebP", "Delta+PNG", "CompressAI", "No-Ref", "Ref-Cond", "Mode Sel"]
    header_h, row_h = 86, 88
    draw.rectangle((x0, y0, x0 + sum(widths), y0 + header_h), fill="#eaf0f4")
    x = x0
    for width, header in zip(widths, headers):
        draw.multiline_text((x + width / 2, y0 + header_h / 2), textwrap.fill(header, 15), anchor="mm", align="center", fill="#142330", font=header_f, spacing=3)
        x += width
    values = []
    for row in rows:
        vals = [row["bucket"], row["JPEG"], row["WebP"], row["Delta+PNG"], row["CompressAI"], row["No-Ref"], row["Ref-Cond"], row["Mode Sel"]]
        values.append(vals)
    for i, vals in enumerate(values):
        y1 = y0 + header_h + i * row_h
        fill = "#f7f9fa" if i % 2 == 0 else "white"
        draw.rectangle((x0, y1, x0 + sum(widths), y1 + row_h), fill=fill, outline="#d4dde3", width=1)
        numeric = []
        for text in vals[1:]:
            try:
                numeric.append(float(text.split()[0]))
            except (ValueError, IndexError):
                numeric.append(float("inf"))
        min_value = min(numeric)
        x = x0
        for j, (width, value) in enumerate(zip(widths, vals)):
            f = body_f
            color = "#17232d"
            if j == 0:
                f = font(20, True)
                draw.text((x + 14, y1 + row_h / 2), value, anchor="lm", fill=color, font=f)
            else:
                if numeric[j - 1] == min_value:
                    f = font(21, True)
                    color = "#0d6b49"
                draw.multiline_text((x + width / 2, y1 + row_h / 2), value, anchor="mm", align="center", fill=color, font=f, spacing=2)
            x += width
    image.save(path)


def chart_axes(title: str, subtitle: str = ""):
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((72, 42), title, fill="#142330", font=font(38, True))
    if subtitle:
        draw.text((74, 100), subtitle, fill="#445663", font=font(19))
    return image, draw


def save_loss_plot(path: Path, histories: dict[str, dict]) -> None:
    image, draw = chart_axes("Training Curves", "Epochs 1–28; average of logged training-loss samples per epoch; loss on log scale; 128px crops, batch 4, lambda 0.0130")
    left, top, right, bottom = 135, 165, 1510, 730
    hseries = {}
    for name, obj in histories.items():
        groups = defaultdict(list)
        for item in obj["history"]:
            if "loss" in item:
                groups[int(item["epoch"])].append(float(item["loss"]))
        hseries[name] = [(epoch + 1, mean(vals)) for epoch, vals in sorted(groups.items())]
    all_losses = [v for series in hseries.values() for _, v in series if v and v > 0]
    low, high = math.log10(min(all_losses)), math.log10(max(all_losses))
    if high - low < .1:
        high = low + .1
    for tick in range(7):
        value = low + (high - low) * tick / 6
        y = bottom - (value - low) / (high - low) * (bottom - top)
        draw.line((left, y, right, y), fill="#e3e9ed", width=2)
        draw.text((left - 16, y), f"10^{value:.1f}", anchor="rm", fill="#445663", font=font(18))
    for epoch in range(1, 29, 2):
        x = left + (epoch - 1) / 27 * (right - left)
        draw.line((x, top, x, bottom), fill="#f0f3f5", width=1)
        draw.text((x, bottom + 16), str(epoch), anchor="ma", fill="#445663", font=font(18))
    series_colors = {"No-Reference": "#276fbf", "Reference-Conditioned": "#17824e"}
    for name, series in hseries.items():
        points = [(left + (epoch - 1) / 27 * (right - left), bottom - (math.log10(loss) - low) / (high - low) * (bottom - top)) for epoch, loss in series if loss > 0]
        draw.line(points, fill=series_colors[name], width=5)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=series_colors[name])
    draw.line((left, top, left, bottom), fill="#253744", width=3)
    draw.line((left, bottom, right, bottom), fill="#253744", width=3)
    draw.text(((left + right) / 2, bottom + 47), "Epoch (zero-based metrics displayed as 1–28)", anchor="ma", fill="#233746", font=font(21))
    # Legend
    for i, (name, color) in enumerate(series_colors.items()):
        x = 570 + i * 390
        draw.line((x, 832, x + 62, 832), fill=color, width=6)
        draw.text((x + 76, 832), name, anchor="lm", fill="#233746", font=font(22, True))
    draw.rounded_rectangle((1120, 150, 1510, 220), radius=6, fill="#fff4d8", outline="#d6ad56", width=2)
    draw.multiline_text((1138, 163), "validation not logged\nno resume marker in metrics", fill="#664b16", font=font(18, True), spacing=4)
    image.save(path)


def save_bpp_plot(path: Path, main_data: list[dict], settings: list[dict]) -> None:
    image, draw = chart_axes("Bits per Pixel by Degree of Change", "Nearest finite-PSNR quality setting to Ref-Cond mean; neural bpp estimated from likelihoods")
    left, top, right, bottom = 125, 190, 1510, 720
    numeric = []
    for row in main_data:
        for method in METHODS:
            value = row["raw_values"].get(method)
            if value is not None:
                numeric.append(value)
    ymax = max(numeric) * 1.12
    for i in range(6):
        v = ymax * i / 5
        y = bottom - v / ymax * (bottom - top)
        draw.line((left, y, right, y), fill="#e3e9ed", width=2)
        draw.text((left - 14, y), f"{v:.2f}", anchor="rm", fill="#445663", font=font(18))
    group_w = (right - left) / len(BUCKETS)
    bar_w = 25
    gap = 8
    span = len(METHODS) * (bar_w + gap) - gap
    for bi, bucket in enumerate(BUCKETS):
        row = next(x for x in main_data if x["bucket"] == bucket)
        center = left + (bi + .5) * group_w
        start = center - span / 2
        for mi, method in enumerate(METHODS):
            val = row["raw_values"].get(method)
            x = start + mi * (bar_w + gap)
            if val is None:
                draw.text((x + bar_w/2, bottom - 10), "×", anchor="mm", fill=COLORS[mi], font=font(20, True))
                continue
            y = bottom - val / ymax * (bottom - top)
            draw.rectangle((x, y, x + bar_w, bottom), fill=COLORS[mi])
        draw.text((center, bottom + 27), bucket, anchor="ma", fill="#233746", font=font(21, True))
    draw.line((left, top, left, bottom), fill="#253744", width=3)
    draw.line((left, bottom, right, bottom), fill="#253744", width=3)
    for i, method in enumerate(METHODS):
        x = 150 + (i % 4) * 350
        y = 800 + (i // 4) * 35
        draw.rectangle((x, y - 10, x + 20, y + 10), fill=COLORS[i])
        draw.text((x + 30, y), method, anchor="lm", fill="#233746", font=font(21, True))
    image.save(path)


def save_mode_plot(path: Path, mode_rows: list[dict]) -> None:
    rows = [x for x in mode_rows if x["degree_of_change"] == "unrelated"]
    mean_mode_bpp = mean([x["mode_selection"]["estimated_bpp_from_likelihoods"] for x in rows])
    mean_noref_bpp = mean([x["noref"]["estimated_bpp_from_likelihoods"] for x in rows])
    mean_mode_j = mean([x["mode_selection"]["objective_j"] for x in rows])
    mean_noref_j = mean([x["mode_selection"]["noref_objective_j"] for x in rows])
    worse = sum(x["mode_selection"]["objective_j"] > x["mode_selection"]["noref_objective_j"] for x in rows)
    image, draw = chart_axes("Mode Selection Check · Unrelated Bucket", "100 evaluation pairs; objective J is the stored mode-selector objective")
    values = [("Mean estimated bpp", mean_noref_bpp, mean_mode_bpp), ("Mean J", mean_noref_j, mean_mode_j)]
    colors = ("#276fbf", "#bc4b68")
    xcenters = (510, 1090)
    for (label, a, b), center in zip(values, xcenters):
        ymax = max(a, b) * 1.25
        ybase, ytop = 650, 245
        draw.text((center, 190), label, anchor="mm", fill="#142330", font=font(28, True))
        for j, value in enumerate((a, b)):
            x = center - 100 + j * 120
            y = ybase - value / ymax * (ybase - ytop)
            draw.rectangle((x, y, x + 76, ybase), fill=colors[j])
            draw.text((x + 38, y - 15), f"{value:.6f}", anchor="ms", fill="#233746", font=font(20, True))
        draw.text((center - 62, ybase + 22), "No-Ref", anchor="ma", fill="#233746", font=font(19))
        draw.text((center + 60, ybase + 22), "Mode Sel", anchor="ma", fill="#233746", font=font(19))
    draw.rounded_rectangle((260, 750, 1340, 840), radius=8, fill="#e8f5ee", outline="#6da88a", width=2)
    draw.text((800, 795), f"PASS · Mode Selection worse on {worse}/100 pairs by J (must be 0)", anchor="mm", fill="#145b3d", font=font(24, True))
    image.save(path)


def save_reconstruction_placeholder(path: Path, selected: list[dict]) -> None:
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 38), "Reconstructions by Degree of Change", fill="#142330", font=font(36, True))
    draw.text((62, 92), "Placeholder only: source reference/target PNGs are not present inside results/.", fill="#a33737", font=font(20, True))
    cols = ("Reference", "Target", "No-Ref decode", "Ref-Cond decode", "Mode Selection choice")
    left, top = 46, 145
    cell_w, cell_h, row_h = 300, 98, 122
    for i, title in enumerate(cols):
        x = left + i * cell_w
        draw.rectangle((x, top, x + cell_w - 8, top + 44), fill="#eaf0f4")
        draw.text((x + (cell_w - 8)/2, top + 22), title, anchor="mm", fill="#142330", font=font(17, True))
    for ri, row in enumerate(selected):
        y = top + 53 + ri * row_h
        draw.text((left, y + 18), f"{row['bucket']} · {row['pair_id']}", fill="#233746", font=font(16, True))
        for ci in range(5):
            x = left + ci * cell_w
            ybox = y + 44
            draw.rectangle((x, ybox, x + cell_w - 8, ybox + cell_h - 12), fill="#f6f8f9", outline="#c9d3da", width=2)
            label = "missing source image" if ci < 2 else "epoch-27 decode exists"
            draw.multiline_text((x + (cell_w-8)/2, ybox + (cell_h-12)/2), label, anchor="mm", align="center", fill="#697985", font=font(15), spacing=4)
    footer = "No image is fabricated. Use make_reconstructions.ps1 to load original pair PNGs from the local test dataset and pair them with existing evaluation decodes. Full per-cell paths will be written to reconstruction_sources.csv."
    draw.multiline_text((58, 825), textwrap.fill(footer, 142), fill="#445663", font=font(15), spacing=3)
    image.save(path)


def save_combined(path: Path, panel_paths: list[Path]) -> None:
    canvas = Image.new("RGB", (1600, 900), "white")
    for index, panel_path in enumerate(panel_paths):
        panel = Image.open(panel_path).convert("RGB").resize((800, 450), Image.Resampling.LANCZOS)
        canvas.paste(panel, ((index % 2) * 800, (index // 2) * 450))
    canvas.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eval_rows = read_jsonl(RESULTS / "pipeline_eval_modes.jsonl")
    classical_rows = read_jsonl(RESULTS / "baselines_classical_full_test.jsonl")
    compressai_runs = read_jsonl(RESULTS / "baselines_compressai_full_test.jsonl")
    classical = {row["pair_id"]: row for row in classical_rows}
    eval_by_id = {row["pair_id"]: row for row in eval_rows}
    if len(eval_by_id) != len(eval_rows):
        raise ValueError("duplicate IDs in evaluation file")
    if len(classical) != len(classical_rows):
        raise ValueError("duplicate IDs in full-test classical baseline")
    if len(classical_rows) != 2000 or Counter(x["degree_of_change"] for x in classical_rows) != Counter({b: 400 for b in BUCKETS}):
        raise ValueError("full-test classical baseline is not 2,000 rows / 400 per bucket")
    if len(eval_rows) != 500 or Counter(x["degree_of_change"] for x in eval_rows) != Counter({b: 100 for b in BUCKETS}):
        raise ValueError("pipeline evaluation is not the expected 500 pairs / 100 per bucket")
    eval_ids = set(eval_by_id)
    missing_classical = sorted(eval_ids - set(classical))
    if missing_classical:
        raise ValueError(f"{len(missing_classical)} evaluation IDs missing from classical file")
    bucket_mismatches = [pid for pid in eval_ids if eval_by_id[pid]["degree_of_change"] != classical[pid]["degree_of_change"]]
    if bucket_mismatches:
        raise ValueError(f"bucket mismatch for IDs: {bucket_mismatches[:5]}")
    ca = {}
    for run in compressai_runs:
        quality = int(run["quality"])
        results = run["results"]
        if len(results) != 2000:
            raise ValueError(f"CompressAI q{quality} has {len(results)} rather than 2,000 rows")
        by_id = {x["pair_id"]: x for x in results}
        if len(by_id) != len(results):
            raise ValueError(f"duplicate IDs in CompressAI q{quality}")
        if set(by_id) != set(classical):
            raise ValueError(f"CompressAI q{quality} IDs differ from the full-test classical set")
        if any(by_id[pid]["degree_of_change"] != classical[pid]["degree_of_change"] for pid in by_id):
            raise ValueError(f"bucket labels differ between CompressAI q{quality} and classical set")
        ca[quality] = by_id
    if set(ca) != {1, 3, 5, 7}:
        raise ValueError(f"unexpected CompressAI qualities: {sorted(ca)}")
    missing_ca = {quality: sorted(eval_ids - set(by_id)) for quality, by_id in ca.items()}
    if any(missing_ca.values()):
        raise ValueError(f"evaluation IDs missing from CompressAI qualities: {missing_ca}")
    for row in eval_rows:
        for method in ("noref", "reference_conditioned", "mode_selection"):
            for field in ("estimated_bpp_from_likelihoods", "mse", "psnr_db", "ms_ssim"):
                value = float(row[method][field])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {method}.{field} for {row['pair_id']}")
            if float(row[method]["estimated_bpp_from_likelihoods"]) <= 0:
                raise ValueError(f"degenerate neural bpp for {row['pair_id']}")
    for row in classical_rows:
        for codec in ("jpeg", "webp"):
            for setting in row[codec]:
                if not math.isfinite(float(setting["bits_per_pixel"])) or math.isnan(float(setting["psnr_db"])):
                    raise ValueError(f"invalid {codec} metric for {row['pair_id']}")
        if not math.isfinite(float(row["delta_png"]["bits_per_pixel"])):
            raise ValueError(f"invalid Delta+PNG bpp for {row['pair_id']}")

    # The baseline records contain per-image dB values. A bucket-setting mean is +inf if any source value is +inf.
    selection: dict[tuple[str, str], dict] = {}
    quality_detail = []
    for bucket in BUCKETS:
        subset = [row for row in eval_rows if row["degree_of_change"] == bucket]
        target_psnr = mean([float(row["reference_conditioned"]["psnr_db"]) for row in subset])
        for codec, field in (("JPEG", "jpeg"), ("WebP", "webp")):
            qualities = sorted({int(v["quality"]) for pid in eval_ids for v in classical[pid][field]})
            candidates = []
            for q in qualities:
                vals = [float(next(v for v in classical[row["pair_id"]][field] if int(v["quality"]) == q)["psnr_db"]) for row in subset]
                psnr = mean(vals)
                bpp = mean([float(next(v for v in classical[row["pair_id"]][field] if int(v["quality"]) == q)["bits_per_pixel"]) for row in subset])
                detail = {"bucket": bucket, "codec": codec, "quality": q, "pair_count": len(subset), "mean_bpp": bpp, "mean_psnr_db": psnr, "infinite_psnr_count": sum(math.isinf(v) for v in vals), "selected": False, "psnr_gap_from_refcond_db": None}
                quality_detail.append(detail)
                if math.isfinite(psnr):
                    candidates.append((abs(psnr - target_psnr), q, psnr, bpp))
            best = min(candidates, default=None)
            selection[(bucket, codec)] = {"quality": best[1] if best else None, "mean_psnr": best[2] if best else None, "mean_bpp": best[3] if best else None, "ref_psnr": target_psnr}
        candidates = []
        for q, by_id in sorted(ca.items()):
            vals = [float(by_id[row["pair_id"]]["psnr_db"]) for row in subset]
            psnr = mean(vals)
            bpp = mean([float(by_id[row["pair_id"]]["bpp"]) for row in subset])
            detail = {"bucket": bucket, "codec": "CompressAI", "quality": q, "pair_count": len(subset), "mean_bpp": bpp, "mean_psnr_db": psnr, "infinite_psnr_count": sum(math.isinf(v) for v in vals), "selected": False, "psnr_gap_from_refcond_db": None}
            quality_detail.append(detail)
            if math.isfinite(psnr):
                candidates.append((abs(psnr-target_psnr),q,psnr,bpp))
        best = min(candidates, default=None)
        selection[(bucket, "CompressAI")] = {"quality": best[1] if best else None, "mean_psnr": best[2] if best else None, "mean_bpp": best[3] if best else None, "ref_psnr": target_psnr}
    for detail in quality_detail:
        best = selection[(detail["bucket"], detail["codec"])]["quality"]
        detail["selected"] = best is not None and int(detail["quality"]) == best
        detail["psnr_gap_from_refcond_db"] = (detail["mean_psnr_db"] - selection[(detail["bucket"], detail["codec"])]["ref_psnr"]) if math.isfinite(detail["mean_psnr_db"]) else None

    main_data = []
    main_csv = []
    metric_csv = []
    for bucket in BUCKETS:
        subset = [row for row in eval_rows if row["degree_of_change"] == bucket]
        ids = [row["pair_id"] for row in subset]
        chosen = {}
        for codec in ("JPEG", "WebP", "CompressAI"):
            q = selection[(bucket, codec)]["quality"]
            chosen[codec] = None
            if q is not None and codec in ("JPEG", "WebP"):
                field = codec.lower()
                chosen[codec] = mean([float(next(v for v in classical[pid][field] if int(v["quality"]) == q)["bits_per_pixel"]) for pid in ids])
            elif q is not None:
                chosen[codec] = mean([float(ca[q][pid]["bpp"]) for pid in ids])
        delta = mean([float(classical[pid]["delta_png"]["bits_per_pixel"]) for pid in ids])
        noref = mean([float(row["noref"]["estimated_bpp_from_likelihoods"]) for row in subset])
        ref = mean([float(row["reference_conditioned"]["estimated_bpp_from_likelihoods"]) for row in subset])
        mode = mean([float(row["mode_selection"]["estimated_bpp_from_likelihoods"]) for row in subset])
        values = {"JPEG": chosen["JPEG"], "WebP": chosen["WebP"], "Delta+PNG": delta, "CompressAI": chosen["CompressAI"], "No-Ref": noref, "Ref-Cond": ref, "Mode Sel": mode}
        display = {k: ("missing" if values[k] is None else f"{values[k]:.5f}") for k in values}
        for key in ("JPEG", "WebP", "CompressAI"):
            q = selection[(bucket, key)]["quality"]
            if values[key] is not None:
                display[key] += f" (q{q})"
        display["Delta+PNG"] += " (lossless)"
        main_data.append({"bucket": bucket, "raw_values": values, **display})
        main_csv.append({"bucket": bucket, "pairs": len(subset), "JPEG": fmt(values["JPEG"]), "JPEG_quality": selection[(bucket,"JPEG")]["quality"] or "missing", "WebP": fmt(values["WebP"]), "WebP_quality": selection[(bucket,"WebP")]["quality"] or "missing", "Delta+PNG": fmt(delta), "Delta+PNG_quality": "lossless", "CompressAI": fmt(values["CompressAI"]), "CompressAI_quality": selection[(bucket,"CompressAI")]["quality"] or "missing", "No-Ref_estimated": fmt(noref), "Ref-Cond_estimated": fmt(ref), "Mode_Sel_estimated": fmt(mode)})
        # Distortion-quality companion for the displayed/nearest settings.
        for codec in ("JPEG", "WebP", "CompressAI"):
            q = selection[(bucket, codec)]["quality"]
            if q is None:
                metric_csv.append({"bucket": bucket, "method": codec, "setting": "missing", "pairs": len(subset), "mean_psnr_db": "missing", "mean_ms_ssim": "missing", "note": "No finite mean PSNR setting available for matching."})
            elif codec == "CompressAI":
                metric_csv.append({"bucket": bucket, "method": codec, "setting": f"q{q}", "pairs": len(subset), "mean_psnr_db": fmt(selection[(bucket,codec)]["mean_psnr"]), "mean_ms_ssim": "missing", "note": "MS-SSIM not stored in CompressAI baseline file."})
            else:
                field=codec.lower()
                vals=[float(next(v for v in classical[pid][field] if int(v["quality"])==q)["psnr_db"]) for pid in ids]
                metric_csv.append({"bucket":bucket,"method":codec,"setting":f"q{q}","pairs":len(subset),"mean_psnr_db":fmt(mean(vals)),"mean_ms_ssim":"missing","note":"MS-SSIM not stored in classical baseline file."})
        for method, source_key in (("No-Ref","noref"),("Ref-Cond","reference_conditioned"),("Mode Sel","mode_selection")):
            metric_csv.append({"bucket":bucket,"method":method,"setting":"lambda=0.0130","pairs":len(subset),"mean_psnr_db":fmt(mean([float(x[source_key]["psnr_db"]) for x in subset])),"mean_ms_ssim":fmt(mean([float(x[source_key]["ms_ssim"]) for x in subset])),"note":"Neural likelihood-bpp evaluation."})
        metric_csv.append({"bucket":bucket,"method":"Delta+PNG","setting":"lossless","pairs":len(subset),"mean_psnr_db":"lossless","mean_ms_ssim":"lossless","note":"Do not compare distortion metrics; exact reconstruction path."})

    write_csv(OUT / "results_bpp_table.csv", main_csv)
    write_csv(OUT / "results_psnr_msssim_table.csv", metric_csv)
    comparison_checks = []
    for bucket in BUCKETS:
        values = next(x["raw_values"] for x in main_data if x["bucket"] == bucket)
        for baseline in ("JPEG", "WebP"):
            q = selection[(bucket, baseline)]["quality"]
            psnr_gap = (selection[(bucket, baseline)]["ref_psnr"] - selection[(bucket, baseline)]["mean_psnr"]) if q is not None else None
            for neural in ("No-Ref", "Ref-Cond", "Mode Sel"):
                neural_bpp = values[neural]
                baseline_bpp = values[baseline]
                comparison_checks.append({"bucket":bucket,"neural_method":neural,"neural_bpp_estimated":fmt(neural_bpp),"baseline":baseline,"baseline_quality":f"q{q}" if q is not None else "missing","baseline_bpp":fmt(baseline_bpp),"neural_bpp_lower":("yes" if neural_bpp < baseline_bpp else "no") if baseline_bpp is not None else "missing","baseline_minus_refcond_psnr_db":fmt(-psnr_gap if psnr_gap is not None else None),"note":"Rate-only comparison; PSNR gap means this is not asserted as matched-quality superiority."})
    write_csv(OUT / "baseline_comparison_checks.csv", comparison_checks)
    write_csv(OUT / "quality_settings_500_pairs.csv", [{**x, "mean_bpp":fmt(x["mean_bpp"]), "mean_psnr_db":fmt(x["mean_psnr_db"]), "psnr_gap_from_refcond_db":fmt(x["psnr_gap_from_refcond_db"])} for x in quality_detail])
    with (OUT / "results_bpp_table.md").open("w",encoding="utf-8") as handle:
        handle.write("# Results: Bits per Pixel by Degree of Change\n\n")
        handle.write("| Bucket | Pairs | JPEG | WebP | Delta+PNG | CompressAI | No-Ref | Ref-Cond | Mode Sel |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in main_data:
            handle.write(f"| {row['bucket']} | 100 | {row['JPEG']} | {row['WebP']} | {row['Delta+PNG']} | {row['CompressAI']} | {row['No-Ref']} | {row['Ref-Cond']} | {row['Mode Sel']} |\n")
        handle.write(f"\n{FOOTNOTE}\n\n")
        handle.write("Baseline settings are selected per bucket by nearest finite mean PSNR to Ref-Cond; exact selected PSNRs/gaps and all settings are in `quality_settings_500_pairs.md`. `missing` means no finite mean-PSNR quality setting exists, not that the baseline run was absent. Delta+PNG is lossless.\n")
    save_table_png(OUT / "results_bpp_table.png", main_data, quality_detail)

    # Full-test baseline summary is separate from the 500-pair matched table.
    full_rows = []
    full_q_rows = []
    for bucket in ("overall_2000", *BUCKETS):
        selected = classical_rows if bucket == "overall_2000" else [x for x in classical_rows if x["degree_of_change"] == bucket]
        selected_ids = {x["pair_id"] for x in selected}
        for codec, field in (("JPEG","jpeg"),("WebP","webp")):
            qualities=sorted({int(v["quality"]) for x in selected for v in x[field]})
            for q in qualities:
                vals=[next(v for v in x[field] if int(v["quality"])==q) for x in selected]
                ps=[float(v["psnr_db"]) for v in vals]
                row={"scope":"full test: 2,000 total; 400/bucket" if bucket=="overall_2000" else "full test bucket","bucket":bucket,"method":codec,"setting":f"q{q}","pairs":len(selected),"mean_bpp":mean([float(v["bits_per_pixel"]) for v in vals]),"mean_psnr_db":mean(ps),"infinite_psnr_count":sum(math.isinf(v) for v in ps),"mean_ms_ssim":"missing"}
                full_rows.append(row); full_q_rows.append({**row,"mean_bpp":fmt(row["mean_bpp"]),"mean_psnr_db":fmt(row["mean_psnr_db"])})
        vals=[x["delta_png"] for x in selected]
        full_rows.append({"scope":"full test: 2,000 total; 400/bucket" if bucket=="overall_2000" else "full test bucket","bucket":bucket,"method":"Delta+PNG","setting":"lossless","pairs":len(selected),"mean_bpp":mean([float(v["bits_per_pixel"]) for v in vals]),"mean_psnr_db":"lossless","infinite_psnr_count":0,"mean_ms_ssim":"lossless"})
        full_q_rows.append({**full_rows[-1],"mean_bpp":fmt(full_rows[-1]["mean_bpp"])})
        for q, by_id in sorted(ca.items()):
            vals=[by_id[pid] for pid in selected_ids]
            ps=[float(v["psnr_db"]) for v in vals]
            row={"scope":"full test: 2,000 total; 400/bucket" if bucket=="overall_2000" else "full test bucket","bucket":bucket,"method":"CompressAI","setting":f"q{q}","pairs":len(vals),"mean_bpp":mean([float(v["bpp"]) for v in vals]),"mean_psnr_db":mean(ps),"infinite_psnr_count":sum(math.isinf(v) for v in ps),"mean_ms_ssim":"missing"}
            full_rows.append(row); full_q_rows.append({**row,"mean_bpp":fmt(row["mean_bpp"]),"mean_psnr_db":fmt(row["mean_psnr_db"])})
    write_csv(OUT / "baselines_all_2000.csv", full_q_rows)
    with (OUT / "baselines_all_2000.md").open("w",encoding="utf-8") as handle:
        handle.write("# Full-Test Baselines: 2,000 Pairs\n\nThis table is **separate** from the 500-pair matched neural evaluation. It is aggregated directly from the full-test baseline files. `mean_ms_ssim` is missing because these baseline files do not store MS-SSIM. `+inf` means one or more per-image PSNR values were infinite.\n\n")
        handle.write("| Scope | Bucket | Method | Setting | Pairs | Mean bpp | Mean PSNR (dB) | +inf PSNR rows | Mean MS-SSIM |\n|---|---|---|---|---:|---:|---:|---:|---|\n")
        for row in full_q_rows:
            handle.write(f"| {row['scope']} | {row['bucket']} | {row['method']} | {row['setting']} | {row['pairs']} | {row['mean_bpp']} | {row.get('mean_psnr_db','lossless')} | {row['infinite_psnr_count']} | {row['mean_ms_ssim']} |\n")

    with (OUT / "quality_settings_500_pairs.md").open("w",encoding="utf-8") as handle:
        handle.write("# All Baseline Settings on the Same 500 Pairs\n\nQuality-matching target is each bucket's mean Ref-Cond PSNR. Means include all 100 logged per-image PSNR values; if any are `+inf`, the arithmetic mean is `+inf`. No finite quality is selected where every setting has infinite mean PSNR.\n\n")
        handle.write("| Bucket | Codec | Setting | Pairs | Mean bpp | Mean PSNR dB | Infinite PSNR rows | Selected | PSNR gap to Ref-Cond dB |\n|---|---|---:|---:|---:|---:|---:|---|---:|\n")
        for x in quality_detail:
            handle.write(f"| {x['bucket']} | {x['codec']} | q{x['quality']} | {x['pair_count']} | {fmt(x['mean_bpp'])} | {fmt(x['mean_psnr_db'])} | {x['infinite_psnr_count']} | {'yes' if x['selected'] else 'no'} | {fmt(x['psnr_gap_from_refcond_db'])} |\n")
        handle.write("\nDelta+PNG is lossless and is listed in the primary bpp table, not in the quality sweep.\n")

    # Mode-selection correctness from per-pair records, not the stored summary PASS string.
    unrelated = [x for x in eval_rows if x["degree_of_change"] == "unrelated"]
    mode_bpp = mean([float(x["mode_selection"]["estimated_bpp_from_likelihoods"]) for x in unrelated])
    noref_bpp = mean([float(x["noref"]["estimated_bpp_from_likelihoods"]) for x in unrelated])
    mode_j = mean([float(x["mode_selection"]["objective_j"]) for x in unrelated])
    noref_j = mean([float(x["mode_selection"]["noref_objective_j"]) for x in unrelated])
    worse = sum(float(x["mode_selection"]["objective_j"]) > float(x["mode_selection"]["noref_objective_j"]) for x in unrelated)
    mode_status = "PASS" if worse == 0 else "FAIL"
    (OUT / "mode_selection_check.txt").write_text(f"{mode_status}: unrelated bucket, n={len(unrelated)}; Mode Selection mean estimated bpp={mode_bpp:.8f}, mean J={mode_j:.10f}; No-Reference mean estimated bpp={noref_bpp:.8f}, mean J={noref_j:.10f}; Mode Selection worse by J on {worse}/{len(unrelated)} pairs. PASS iff worse count is 0.\n",encoding="utf-8")
    save_mode_plot(OUT / "mode_selection_check.png", eval_rows)

    # Training histories contain only sampled training loss; no validation or resume marker is present.
    noref_metrics = json.loads((RESULTS / "pipeline_noref_lambda_00130_metrics.json").read_text(encoding="utf-8"))
    ref_metrics = json.loads((RESULTS / "pipeline_reference_lambda_00130_metrics.json").read_text(encoding="utf-8"))
    for name, obj in (("No-Reference",noref_metrics),("Reference-Conditioned",ref_metrics)):
        if not obj.get("history") or "validation_loss" in obj or any("val_loss" in row for row in obj["history"]):
            raise ValueError(f"unexpected or empty metrics history for {name}; inspect before plotting")
        if any(not math.isfinite(float(item["loss"])) for item in obj["history"] if "loss" in item):
            raise ValueError(f"non-finite loss value for {name}")
    save_loss_plot(OUT / "loss_curves.png", {"No-Reference":noref_metrics,"Reference-Conditioned":ref_metrics})
    save_bpp_plot(OUT / "bpp_by_bucket.png", main_data, quality_detail)

    # Pick one exact evaluation ID per bucket with both saved reconstructions present.
    reconstruction_rows = []
    for bucket in BUCKETS:
        candidates = [x for x in eval_rows if x["degree_of_change"] == bucket]
        chosen = next((x for x in candidates if (ROOT / x["noref_reconstruction"]).is_file() and (ROOT / x["reference_reconstruction"]).is_file()), None)
        mode_codec=chosen["mode_selection"]["codec"] if chosen else "missing"
        mode_source=(chosen["noref_reconstruction"] if mode_codec=="noref" else chosen["reference_reconstruction"]) if chosen else "missing"
        reconstruction_rows.append({"bucket":bucket,"pair_id":chosen["pair_id"] if chosen else "missing","source_reference":"missing (not in results/)","source_target":"missing (not in results/)","mode_selected_codec":mode_codec,"noref_reconstruction":chosen["noref_reconstruction"] if chosen else "missing","reference_reconstruction":chosen["reference_reconstruction"] if chosen else "missing","mode_selection_decode_source":mode_source,"note":"Cached evaluation decoder image; checkpoint path is not embedded in the evaluation row."})
    write_csv(OUT / "reconstruction_sources.csv", reconstruction_rows)
    # Inputs are intentionally not loaded here; this session may read results/ only.
    save_reconstruction_placeholder(OUT / "reconstructions.png", reconstruction_rows)

    # Allowed claims and a cautious status line derived from the same per-pair rows.
    claims = ["# Claims Allowed\n", "Only statements supported by the stored 500-pair evaluation and the selected full-test baselines are listed here. Neural rates are likelihood estimates, not actual bitstreams.\n", "## Supported\n"]
    rate_diffs=[]
    for bucket in BUCKETS:
        subset=[x for x in eval_rows if x["degree_of_change"]==bucket]
        n=mean([float(x["noref"]["estimated_bpp_from_likelihoods"]) for x in subset])
        r=mean([float(x["reference_conditioned"]["estimated_bpp_from_likelihoods"]) for x in subset])
        pn=mean([float(x["noref"]["psnr_db"]) for x in subset]); pr=mean([float(x["reference_conditioned"]["psnr_db"]) for x in subset])
        rate_diffs.append(abs(pr-pn))
        claims.append(f"- In `{bucket}` (n=100), Ref-Cond mean likelihood-bpp is {r:.5f} versus No-Ref {n:.5f}; mean PSNR is {pr:.3f} dB versus {pn:.3f} dB.")
        for baseline in ("JPEG","WebP"):
            q=selection[(bucket,baseline)]["quality"]
            if q is None:
                claims.append(f"- {baseline} quality-matched bpp comparison in `{bucket}` is unavailable: no finite-mean-PSNR setting exists in the source file.")
            else:
                ps=selection[(bucket,baseline)]["mean_psnr"]
                gap=selection[(bucket,baseline)]["ref_psnr"]-ps
                values=main_data[BUCKETS.index(bucket)]["raw_values"]
                claims.append(f"- Nearest finite `{baseline}` setting in `{bucket}` is q{q}: {ps:.3f} dB versus Ref-Cond {selection[(bucket,baseline)]['ref_psnr']:.3f} dB (gap {gap:.3f} dB). Its bpp is {values[baseline]:.5f} versus Ref-Cond {values['Ref-Cond']:.5f}; the quality gap is reported, so this is not described as matched-quality superiority.")
                for neural in ("No-Ref", "Ref-Cond", "Mode Sel"):
                    claims.append(f"  - {neural} bpp {values[neural]:.5f} {'is' if values[neural] < values[baseline] else 'is not'} lower than nearest `{baseline}` q{q} bpp {values[baseline]:.5f}; the closest classical mean is {gap:.3f} dB below Ref-Cond, so do not call this a matched-quality win.")
    claims.append(f"- Mode Selection has mean J {mode_j:.10f} versus No-Ref {noref_j:.10f} in the unrelated bucket; it is worse on {worse}/100 pairs by the stored objective ({mode_status}).")
    claims.append("- The 500-pair evaluation contains 100 pairs per bucket. Full-test classical and generic CompressAI baselines have 2,000 records per configuration; these are a separate population.")
    claims += ["\n## Do Not Claim\n", "- Do not claim BD-rate, a complete rate-distortion curve, or a full 2,000-pair learned-codec evaluation; only one lambda and 500 learned-evaluation pairs are present.\n- Do not claim neural codecs beat JPEG/WebP at matched quality: the nearest available classical means remain substantially below Ref-Cond PSNR, and JPEG has no finite mean-PSNR match for moderate/major.\n- Do not claim generalization to real version-history screenshots from this generated WebSight-derived evaluation set.\n- Do not compare Delta+PNG's lossless rate as if it had lossy PSNR/MSSSIM; its distortion cells are labeled lossless.\n"]
    status_line = (f"On the 500-pair subset, Ref-Cond has lower likelihood-estimated bpp than No-Ref in all five buckets; mean PSNR differs by at most {max(rate_diffs):.3f} dB. Single-lambda subset evidence only; no matched-quality JPEG/WebP superiority, full RD curve, or BD-rate is claimed.")
    claims += ["\n## Replacement Status Line\n", status_line+"\n"]
    (OUT / "claims_allowed.md").write_text("\n".join(claims),encoding="utf-8")

    # Independent arithmetic examples from the raw per-pair JSONL rows.
    arithmetic=[]
    for bucket, method, field in (("identical","No-Ref","noref"),("unrelated","Ref-Cond","reference_conditioned")):
        subset=[x for x in eval_rows if x["degree_of_change"]==bucket]
        vals=[float(x[field]["estimated_bpp_from_likelihoods"]) for x in subset]
        arithmetic.append(f"{bucket} {method} bpp: sum of 100 raw per-pair values = {math.fsum(vals):.12f}; divide by 100 = {math.fsum(vals)/100:.12f}.")
    bucket="minor"; q=selection[(bucket,"WebP")]["quality"]
    subset=[x for x in eval_rows if x["degree_of_change"]==bucket]
    vals=[float(next(v for v in classical[x["pair_id"]]["webp"] if int(v["quality"])==q)["bits_per_pixel"]) for x in subset]
    arithmetic.append(f"{bucket} WebP q{q} bpp: sum of 100 raw matched-ID values = {math.fsum(vals):.12f}; divide by 100 = {math.fsum(vals)/100:.12f}.")

    # Losses: average logged samples per epoch; metrics are 0-based and contain no explicit resume event.
    def loss_summary(obj):
        g=defaultdict(list)
        for x in obj["history"]:
            if "loss" in x:g[int(x["epoch"])].append(float(x["loss"]))
        return {e:mean(v) for e,v in g.items()}
    nloss,rloss=loss_summary(noref_metrics),loss_summary(ref_metrics)
    # Source-level audit numbers and missing fields.
    inf_counts={}
    for bucket in BUCKETS:
        subset=[x for x in eval_rows if x["degree_of_change"]==bucket]
        for codec,field in (("JPEG","jpeg"),("WebP","webp")):
            for q in sorted({int(v["quality"]) for x in subset for v in classical[x["pair_id"]][field]}):
                inf_counts[(bucket,codec,q)]=sum(math.isinf(float(v["psnr_db"])) for x in subset for v in classical[x["pair_id"]][field] if int(v["quality"])==q)
    report=["# Slide Asset Verification Report","",f"Sources: `results/pipeline_eval_modes.jsonl` ({len(eval_rows)} rows), `results/baselines_classical_full_test.jsonl` ({len(classical_rows)} rows), `results/baselines_compressai_full_test.jsonl` ({len(compressai_runs)} quality records), and both pipeline metrics JSON files.","","## Pair and Source Audit",f"- Evaluation pair IDs: {len(eval_ids)} unique; bucket counts: {dict(Counter(x['degree_of_change'] for x in eval_rows))}.",f"- Missing IDs in full-test classical file: {len(missing_classical)}.",f"- Missing IDs in CompressAI q records: { {q:len(v) for q,v in missing_ca.items()} }.",f"- Bucket-label mismatches in the ID join: {len(bucket_mismatches)}.","- Classical full-test baseline: 2,000 rows (400/bucket); CompressAI q1/q3/q5/q7: 2,000 rows each.","- No-Ref, Ref-Cond, and Mode Sel rates in the 500-pair table are estimated from likelihoods; quality settings are selected by nearest finite arithmetic mean PSNR to Ref-Cond within each bucket.","","## Quality Matching", "| Bucket | Ref-Cond mean PSNR | JPEG selected | WebP selected | CompressAI selected |", "|---|---:|---|---|---|"]
    for b in BUCKETS:
        rp=selection[(b,"JPEG")]["ref_psnr"]
        cells=[]
        for codec in ("JPEG","WebP","CompressAI"):
            s=selection[(b,codec)]; cells.append("missing" if s["quality"] is None else f"q{s['quality']} ({s['mean_psnr']:.3f} dB, gap {s['ref_psnr']-s['mean_psnr']:.3f} dB)")
        report.append(f"| {b} | {rp:.3f} | {cells[0]} | {cells[1]} | {cells[2]} |")
    report += ["", "A setting with any +inf source PSNR has +inf arithmetic mean and is not a finite match. JPEG moderate/major therefore have no selected setting. The +inf entries are preserved in `quality_settings_500_pairs.csv` and flagged with per-setting counts; they are not silently removed.","","## Training Curves",f"- No-Reference: {len(noref_metrics['history'])} logged training samples, zero-based epochs {min(nloss)}–{max(nloss)}; validation loss not logged.",f"- Reference-Conditioned: {len(ref_metrics['history'])} logged training samples, zero-based epochs {min(rloss)}–{max(rloss)}; validation loss not logged.","- Metrics JSON contains no resume/interruption marker, so the curve does not annotate a Windows-update interruption. Epochs are plotted 1–28 for readability from zero-based 0–27.","","## Reconstruction Panel", "- Existing neural decode images are in `results/reconstructions/` and are referenced by the 500-row evaluation JSONL. The premade run manifests also record epoch-27 checkpoint provenance.","- Original reference/target PNG files for the selected evaluation IDs are not present in `results/`; they live in the test dataset outside the allowed read scope. Therefore `reconstructions.png` is explicitly a placeholder.","- `make_reconstructions.py` and `make_reconstructions.ps1` are provided to load only the original reference/target PNGs plus already-generated evaluation decodes and assemble the panel; they do not train or load checkpoints.","","## Arithmetic Spot Checks",*arithmetic,"","## Mode Selection",(OUT/"mode_selection_check.txt").read_text(encoding="utf-8").strip(),"","## Rendering and Image Checks","PNG outputs are 1600×900; combined panel is 1600×900. Images are reopened and dimension-checked after writing. Inspect manually before placing them in slides; no external images or network resources were used.","","## Status Line",status_line,"","## Artifact Caveats","- Classical baseline files do not contain MS-SSIM; CompressAI baseline records do not contain MS-SSIM. These companion cells say `missing`.","- Delta+PNG distortion cells say `lossless`, not a numeric PSNR/MS-SSIM comparison.","- Neural results are one lambda, 500 learned pairs, one stored evaluation subset. Do not label them as a full-test neural benchmark or BD-rate."]
    (OUT/"verification_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")

    panel_paths=[OUT/"loss_curves.png",OUT/"bpp_by_bucket.png",OUT/"reconstructions.png",OUT/"mode_selection_check.png"]
    save_combined(OUT/"training_evidence_2x2.png",panel_paths)
    # Open every PNG and fail if dimensions differ from the requested slide canvas.
    for path in sorted(OUT.glob("*.png")):
        with Image.open(path) as check:
            check.verify()
        with Image.open(path) as check:
            if check.size != (1600,900):
                raise ValueError(f"{path.name} is {check.size}, expected 1600x900")
    print(f"Built slide assets in {OUT}; checked {len(list(OUT.glob('*.png')))} PNG files at 1600x900.")


if __name__ == "__main__":
    main()

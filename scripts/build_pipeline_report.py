"""Build a single-file, evidence-only pipeline report from current run artifacts."""
import base64
import argparse
import html
import json
import math
from pathlib import Path


ROOT = Path.cwd()
BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def image_src(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def fmt(value, digits=4):
    if value is None or not isinstance(value, (float, int)) or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def loss_values(path):
    data = read_json(path, {}) or {}
    return [{"step": row.get("step"), "loss": row.get("loss")}
            for row in data.get("history", [])
            if isinstance(row.get("step"), int) and isinstance(row.get("loss"), (int, float))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, help="Add a matched-checkpoint evaluation section")
    parser.add_argument("--matched-eval", type=Path)
    parser.add_argument("--matched-summary", type=Path)
    parser.add_argument("--noref-checkpoint", type=Path)
    parser.add_argument("--ref-checkpoint", type=Path)
    parser.add_argument("--longer-noref-checkpoint", type=Path)
    args = parser.parse_args()

    summary = read_json(ROOT / "results/pipeline_summary.json", {}) or {}
    calibration = read_json(ROOT / "results/pipeline_calibration.json", {}) or {}
    rows = []
    eval_path = ROOT / "results/pipeline_eval_modes.jsonl"
    if eval_path.exists():
        rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    no_ref_loss = loss_values(ROOT / "results/pipeline_noref_lambda_00130_metrics.json")
    ref_loss = loss_values(ROOT / "results/pipeline_reference_lambda_00130_metrics.json")

    table_rows = []
    bucket_summary = summary.get("by_bucket", {})
    for bucket in BUCKETS:
        item = bucket_summary.get(bucket, {})
        classical = item.get("classical", {})
        values = [
            bucket,
            str(item.get("pair_count", 0)),
            fmt((item.get("no_reference") or {}).get("estimated_bpp_from_likelihoods")),
            fmt((item.get("no_reference") or {}).get("psnr_db"), 2),
            fmt((item.get("reference_conditioned") or {}).get("estimated_bpp_from_likelihoods")),
            fmt((item.get("reference_conditioned") or {}).get("psnr_db"), 2),
            fmt((item.get("mode_selection") or {}).get("estimated_bpp_from_likelihoods")),
            fmt((item.get("mode_selection") or {}).get("psnr_db"), 2),
            fmt((classical.get("jpeg_q75") or {}).get("bpp")),
            fmt((classical.get("webp_q75") or {}).get("bpp")),
            fmt((classical.get("delta_png") or {}).get("bpp")),
        ]
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")

    example_sections = []
    for bucket in BUCKETS:
        bucket_rows = [row for row in rows if row.get("degree_of_change") == bucket][:3]
        if not bucket_rows:
            example_sections.append(f"<section><h3>{html.escape(bucket.title())}</h3><p>Reconstructions unavailable.</p></section>")
            continue
        cards = []
        for row in bucket_rows:
            pair_id = row["pair_id"]
            pair_root = ROOT / "data/synthetic_pairs_full/test" / pair_id
            images = [("Reference", image_src(pair_root / "reference.png")),
                      ("Target", image_src(pair_root / "target.png")),
                      ("No-Reference", image_src(row.get("noref_reconstruction", ""))),
                      ("Reference-Conditioned", image_src(row.get("reference_reconstruction", "")))]
            figures = "".join(
                f'<figure><figcaption>{label}</figcaption>{"<img src=" + chr(34) + src + chr(34) + " alt=" + chr(34) + label + chr(34) + ">" if src else "<p>not available</p>"}</figure>'
                for label, src in images
            )
            cards.append(f'<article class="example"><h4>{html.escape(pair_id)}</h4><div class="images">{figures}</div></article>')
        example_sections.append(f"<section><h3>{html.escape(bucket.title())}</h3>{''.join(cards)}</section>")

    cfg = {
        "crop": calibration.get("crop_size"), "pairs": calibration.get("pairs_per_epoch"),
        "epochs": calibration.get("epochs"), "lambda": calibration.get("lambda"),
        "device": calibration.get("device"), "reduced": calibration.get("reduced_scale"),
    }
    config_line = " | ".join(f"{key}={value if value is not None else 'n/a'}" for key, value in cfg.items())
    correctness = summary.get("mode_selection", {})
    comparison_lines = []
    for bucket in ("identical", "minor"):
        item = bucket_summary.get(bucket, {})
        noref_j = (item.get("no_reference") or {}).get("objective_j")
        ref_j = (item.get("reference_conditioned") or {}).get("objective_j")
        if noref_j is None or ref_j is None:
            comparison_lines.append(f"{bucket}: Reference-Conditioned comparison unavailable.")
        elif ref_j < noref_j:
            comparison_lines.append(f"{bucket}: Reference-Conditioned has lower mean J at this single lambda.")
        else:
            comparison_lines.append(f"{bucket}: Reference-Conditioned does not beat No-Reference on mean J at this single lambda.")
    losses_json = json.dumps({"No-Reference": no_ref_loss, "Reference-Conditioned": ref_loss}, allow_nan=False)
    status_color = "var(--green)" if correctness.get("status") == "PASS" else "var(--red)"
    matched_section = ""
    if args.epoch is not None:
        matched_eval_path = args.matched_eval or Path(f"results/matched_epoch_{args.epoch}_modes.jsonl")
        matched_summary_path = args.matched_summary or Path(f"results/matched_epoch_{args.epoch}_summary.json")
        matched_summary = read_json(ROOT / matched_summary_path, {})
        if matched_summary is None or not (ROOT / matched_eval_path).is_file():
            raise FileNotFoundError(f"matched epoch {args.epoch} evaluation/summary artifacts are missing")
        if matched_summary.get("epoch") != args.epoch:
            raise ValueError(f"matched summary epoch {matched_summary.get('epoch')} does not match requested epoch {args.epoch}")
        matched_buckets = matched_summary.get("by_bucket", {})
        matched_rows = []
        for bucket in BUCKETS:
            item = matched_buckets.get(bucket, {})
            classical = item.get("classical") or {}
            comparison_rows = [
                (f"No-Reference epoch {args.epoch}", item.get("no_reference")),
                (f"Reference-Conditioned epoch {args.epoch}", item.get("reference_conditioned")),
                ("Mode Selection", item.get("mode_selection")),
            ]
            longer = item.get("longer_trained_no_reference_not_matched")
            if longer and longer.get("estimated_bpp_from_likelihoods") is not None:
                comparison_rows.append(("No-Reference epoch 27: longer-trained, not matched", longer))
            comparison_rows.extend((
                ("JPEG quality 75", {"estimated_bpp_from_likelihoods": classical.get("jpeg_q75", {}).get("bpp"),
                                     "psnr_db": classical.get("jpeg_q75", {}).get("psnr_db")}),
                ("WebP quality 75", {"estimated_bpp_from_likelihoods": classical.get("webp_q75", {}).get("bpp"),
                                      "psnr_db": classical.get("webp_q75", {}).get("psnr_db")}),
                ("Delta+PNG (lossless)", {"estimated_bpp_from_likelihoods": classical.get("delta_png", {}).get("bpp"),
                                          "psnr_db": classical.get("delta_png", {}).get("psnr_db")}),
            ))
            for label, values in comparison_rows:
                values = values or {}
                matched_rows.append(
                    "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in (
                        bucket, str(item.get("pair_count", 0)), label,
                        fmt(values.get("estimated_bpp_from_likelihoods")), fmt(values.get("psnr_db"), 2),
                        fmt(values.get("ms_ssim"), 4),
                    )) + "</tr>"
                )
        matched_mode = matched_summary.get("mode_selection", {})
        matched_color = "var(--green)" if matched_mode.get("unrelated_correctness") == "PASS" else "var(--red)"
        matched_paths = matched_summary.get("checkpoint_paths") or {}
        matched_section = f'''<section id="matched-epoch"><h2>Matched checkpoint comparison · epoch {args.epoch} ({args.epoch + 1} epochs completed)</h2>
<p class="notice">Stratified held-out test subset: {sum(item.get("pair_count", 0) for item in matched_buckets.values())} pairs total. Epoch indices are zero-based: K={args.epoch} means {args.epoch + 1} epochs completed. Classical baseline records are joined by the same test pair IDs. Neural bpp is estimated from likelihoods, not actual bitstream bytes; classical bpp comes from encoded files.</p>
<p class="notice">Mode Selection unrelated-bucket check: <span class="status" style="background:{matched_color}">{html.escape(str(matched_mode.get('unrelated_correctness', 'NOT RUN')))}</span> {html.escape(str(matched_mode.get('reason', 'No summary available.')))}</p>
<p class="small">No-Reference checkpoint: {html.escape(str(args.noref_checkpoint or matched_paths.get('noref') or 'n/a'))}<br>Reference checkpoint: {html.escape(str(args.ref_checkpoint or matched_paths.get('reference') or 'n/a'))}<br>Longer-trained comparator: {html.escape(str(args.longer_noref_checkpoint or matched_paths.get('longer_noref') or 'not applicable'))}</p>
<div class="table-wrap"><table><thead><tr><th>Bucket</th><th>Pairs</th><th>Comparison (all use the same pairs)</th><th>bpp</th><th>PSNR dB</th><th>MS-SSIM</th></tr></thead><tbody>{''.join(matched_rows)}</tbody></table></div></section>'''
    html_doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeltaShot | Pipeline evidence</title><style>
:root{{color-scheme:light;--ink:#18212b;--muted:#5f6c78;--line:#d8dee5;--paper:#f7f9fb;--blue:#1769aa;--green:#17805d;--red:#b63a38}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
main{{max-width:1240px;margin:auto;padding:32px 24px 64px}}header{{border-bottom:1px solid var(--line);padding:8px 0 24px}}h1{{font-size:32px;margin:0 0 8px}}h2{{margin:32px 0 12px;font-size:22px}}h3{{margin:20px 0 10px;font-size:18px}}p{{color:var(--muted);max-width:900px}}.notice{{border-left:4px solid var(--blue);padding:12px 16px;background:#eaf3fa;max-width:none}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{background:#edf1f5;font-size:12px;color:#43515e}}td:first-child,th:first-child{{text-align:left}}.status{{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:700;color:white;background:{status_color}}}
.examples{{display:grid;grid-template-columns:1fr;gap:10px}}.example{{border-top:1px solid var(--line);padding:10px 0}}.images{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}figure{{margin:0;min-width:0}}figcaption{{font-size:12px;color:var(--muted);margin-bottom:4px}}figure img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:white;border:1px solid var(--line)}}canvas{{width:100%;height:240px;background:white;border:1px solid var(--line)}}.small{{font-size:13px}}code{{background:#edf1f5;padding:2px 4px;border-radius:3px}}@media(max-width:700px){{main{{padding:20px 14px}}h1{{font-size:26px}}.images{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body><main>
<header><h1>DeltaShot | Run evidence</h1><p>DeltaShot tests whether a target UI screenshot can be compressed using an available earlier reference while preserving standalone fallback for unrelated pages. This run is preliminary evidence, not a conclusion: bpp is estimated from model likelihoods, only one lambda is evaluated, and the results are reduced-scale where noted.</p><p class="notice">Mode Selection unrelated-bucket check: <span class="status">{html.escape(str(correctness.get('status', 'NOT RUN')))}</span> {html.escape(str(correctness.get('reason', 'No summary available.')))}</p></header>
{matched_section}
<h2>Per-Bucket Results</h2><p class="small">{html.escape(config_line)}. Neural bpp is estimated from likelihoods; JPEG/WebP are quality 75 and Delta+PNG values are from existing per-pair baseline records. Metrics are means over the evaluated subset. Missing entries are shown as n/a, not imputed.</p>
<div class="table-wrap"><table><thead><tr><th>Change bucket</th><th>Pairs</th><th>No-ref bpp</th><th>No-ref PSNR</th><th>Ref bpp</th><th>Ref PSNR</th><th>Mode bpp</th><th>Mode PSNR</th><th>JPEG bpp</th><th>WebP bpp</th><th>Delta+PNG bpp</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Training Loss</h2><p class="small">Recorded samples only. No curve is drawn for a model without a metrics artifact.</p><canvas id="loss" width="1200" height="260" aria-label="Training loss curve"></canvas>
<h2>Reconstructions</h2><p class="small">Examples are selected from the evaluated test subset. Rows are shown for each degree-of-change bucket.</p><div class="examples">{''.join(example_sections)}</div>
<h2>Status &amp; Caveats</h2><p>Training/evaluation scale: {html.escape(config_line)}. {'REDUCED-SCALE run; do not report as final.' if cfg['reduced'] else 'Full configured scale.'}</p><p>{html.escape(' '.join(comparison_lines))}</p><p>Mode rule: {html.escape(str(correctness.get('rule', 'not available')))} Correctness compares per-pair J in the unrelated bucket and includes the flag bit for reference mode. This is not a multi-lambda BD-rate result. Cross-attention, full benchmark, and final claims remain pending.</p>
<script>const series={losses_json};const c=document.getElementById('loss'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);const colors=['#1769aa','#17805d'];let idx=0;for(const [name,points] of Object.entries(series)){{if(!points.length)continue;const vals=points.map(p=>p.loss).filter(Number.isFinite);if(!vals.length)continue;const lo=Math.min(...vals),hi=Math.max(...vals),span=Math.max(hi-lo,1e-9);ctx.strokeStyle=colors[idx++%colors.length];ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>{{const x=50+i*Math.max(1,(c.width-100)/(points.length-1));const y=c.height-30-(p.loss-lo)/span*(c.height-65);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(name,65,20+idx*18)}}ctx.strokeStyle='#687582';ctx.beginPath();ctx.moveTo(45,15);ctx.lineTo(45,c.height-30);ctx.lineTo(c.width-15,c.height-30);ctx.stroke();</script>
</main></body></html>'''
    (ROOT / "results/demo.html").write_text(html_doc, encoding="utf-8")
    print("Wrote results/demo.html")


if __name__ == "__main__":
    main()

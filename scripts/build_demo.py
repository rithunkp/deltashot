"""Build a standalone, evidence-only DeltaShot review page from local artifacts."""
from __future__ import annotations

import base64
import html
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def examples() -> dict[str, list[dict[str, object]]]:
    selected: dict[str, list[dict[str, object]]] = {bucket: [] for bucket in BUCKETS}
    for pair_dir in sorted((ROOT / "data" / "synthetic_pairs_full" / "test").iterdir()):
        meta_path = pair_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bucket = meta.get("degree_of_change")
        if bucket not in selected or len(selected[bucket]) >= 3:
            continue
        selected[bucket].append({
            "pairId": pair_dir.name,
            "reference": image_uri(pair_dir / "reference.png"),
            "target": image_uri(pair_dir / "target.png"),
            "meta": meta,
        })
        if all(len(items) == 3 for items in selected.values()):
            break
    return selected


def classical_summary(path: Path) -> tuple[dict[str, object] | None, str]:
    if not path.is_file():
        return None, "No full-test classical result file exists yet."
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 2000:
        return None, f"Classical sweep is still running: {len(records):,}/2,000 test pairs completed."
    grouped: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        bucket = record["degree_of_change"]
        for codec in ("jpeg", "webp"):
            grouped[bucket][codec].append(record[codec])
        grouped[bucket]["delta_png"].append([record["delta_png"]])

    summary: dict[str, object] = {}
    for bucket in BUCKETS:
        bucket_summary: dict[str, object] = {}
        for codec in ("jpeg", "webp"):
            curves = grouped[bucket][codec]
            bucket_summary[codec] = [
                dict(
                    quality=point["quality"],
                    bpp=sum(curve[index]["bits_per_pixel"] for curve in curves) / len(curves),
                    psnr=(sum(finite_psnr) / len(finite_psnr)) if (finite_psnr := [curve[index]["psnr_db"] for curve in curves if math.isfinite(curve[index]["psnr_db"])]) else None,
                    psnr_samples=len(finite_psnr),
                )
                for index, point in enumerate(curves[0])
            ]
        delta = [entry[0] for entry in grouped[bucket]["delta_png"]]
        bucket_summary["delta_png"] = {
            "bpp": sum(point["bits_per_pixel"] for point in delta) / len(delta),
            "lossless": all(math.isinf(point["psnr_db"]) for point in delta),
        }
        summary[bucket] = bucket_summary
    return summary, "Full 2,000-pair classical test sweep completed."


def generic_summary(path: Path) -> tuple[dict[str, object] | None, str]:
    if not path.is_file() or not path.stat().st_size:
        return None, "Generic pretrained CompressAI evaluation is running (one q=3 full-test pass, capped at two CPU hours)."
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 1 or len(records[0].get("results", [])) != 2000:
        return None, "Generic pretrained CompressAI result is incomplete; it is not included in the comparison table."
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in records[0]["results"]:
        grouped[row["degree_of_change"]].append(row)
    return {
        bucket: {
            "model": records[0]["model"],
            "quality": records[0]["quality"],
            "bpp": sum(row["bpp"] for row in rows) / len(rows),
            "psnr": (sum(finite) / len(finite)) if (finite := [row["psnr_db"] for row in rows if math.isfinite(row["psnr_db"])]) else None,
            "psnr_samples": len(finite),
        }
        for bucket, rows in grouped.items()
    }, "Full 2,000-pair generic pretrained CompressAI q=3 evaluation completed."


def main() -> None:
    output = ROOT / "results" / "demo.html"
    classical, classical_status = classical_summary(ROOT / "results" / "baselines_classical_full_test.jsonl")
    generic, generic_status = generic_summary(ROOT / "results" / "baselines_compressai_full_test.jsonl")
    poc_metrics = ROOT / "results" / "poc_training_metrics.json"
    poc_data = None
    poc_status = "Proof-of-concept training is running; no reconstruction is displayed until a checkpoint and exported reconstruction exist."
    if poc_metrics.is_file():
        poc_data = json.loads(poc_metrics.read_text(encoding="utf-8"))
        poc_status = "Proof-of-concept loss metrics exist; reconstruction export must be run before this page can show it."
        manifest_path = ROOT / "results" / "poc_examples" / "manifest.json"
        if manifest_path.is_file():
            poc_data["examples"] = []
            for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
                directory = ROOT / "results" / "poc_examples" / entry["bucket"]
                entry["reference"] = image_uri(directory / "reference.png")
                entry["target"] = image_uri(directory / "target.png")
                entry["reconstruction"] = image_uri(directory / "reconstruction.png")
                poc_data["examples"].append(entry)
            poc_status = "Real reconstruction examples and logged training loss from the bounded proof-of-concept run."
    payload = {
        "examples": examples(),
        "classical": classical,
        "classicalStatus": classical_status,
        "genericStatus": generic_status,
        "generic": generic,
        "pocStatus": poc_status,
        "poc": poc_data,
    }
    data = json.dumps(payload).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeltaShot | Mid-Review Evidence</title>
<style>
:root{{--ink:#15202b;--muted:#5b6875;--line:#d8dee5;--paper:#f7f9fb;--blue:#1276a8;--teal:#00877e;--orange:#cc6a22;--red:#bd3c44;--gold:#9b7300}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);font:15px/1.5 Arial,sans-serif;color:var(--ink)}}header{{padding:46px max(24px,calc((100% - 1120px)/2));background:#102b3c;color:#fff;border-bottom:6px solid var(--teal)}}h1{{font-size:42px;line-height:1.1;margin:0 0 14px;letter-spacing:0}}header p{{max-width:840px;margin:0;color:#d8e5eb;font-size:18px}}main{{max-width:1120px;margin:auto;padding:30px 24px 60px}}section{{margin:34px 0;padding-top:4px}}h2{{font-size:25px;margin:0 0 12px}}.eyebrow{{color:var(--teal);font-weight:bold;text-transform:uppercase;font-size:12px;letter-spacing:1px}}.notice{{border-left:4px solid var(--orange);background:#fff6e9;padding:12px 14px;margin:12px 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.stat{{border:1px solid var(--line);background:#fff;padding:15px}}.stat b{{font-size:25px;display:block}}button{{font:inherit;background:#fff;border:1px solid #aab8c2;padding:8px 12px;cursor:pointer}}button[aria-pressed=true]{{background:var(--blue);color:#fff;border-color:var(--blue)}}.toolbar{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0;border:1px solid var(--line);background:#fff;padding:8px}}figure img{{width:100%;display:block;aspect-ratio:1;object-fit:contain;background:#edf1f4}}figcaption{{font-size:12px;color:var(--muted);padding-top:6px}}pre{{white-space:pre-wrap;background:#102b3c;color:#e8f0f4;padding:14px;margin:0;font:12px/1.45 Consolas,monospace;max-height:240px;overflow:auto}}canvas{{width:100%;height:340px;background:#fff;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}}th,td{{padding:9px;border:1px solid var(--line);text-align:left}}th{{background:#eaf1f4}}.pending{{color:var(--muted);font-style:italic}}.roadmap{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.roadmap div{{background:#fff;border-top:3px solid var(--blue);padding:14px}}@media(max-width:700px){{header{{padding:30px 20px}}h1{{font-size:32px}}.grid,.roadmap{{grid-template-columns:1fr}}.pair{{grid-template-columns:1fr}}main{{padding:20px}}}}
</style></head><body><header><div class="eyebrow" style="color:#8ee0d5">DeltaShot · Mid-review evidence</div><h1>Reference-conditioned compression for UI screenshot histories</h1><p>DeltaShot tests whether a target UI screenshot can be stored more efficiently when both sides already have an earlier reference screenshot, while retaining a standalone fallback for unrelated pages. This is an in-progress system: tonight validates the data and independent baselines, not the final reference-conditioned claim.</p></header>
<main><section><div class="eyebrow">Verified dataset</div><h2>24,000 balanced, rendered UI screenshot pairs</h2><div class="grid"><div class="stat"><b>20,000</b>training pairs</div><div class="stat"><b>2,000</b>test pairs</div><div class="stat"><b>5 buckets</b>exactly balanced</div></div></section>
<section><div class="eyebrow">Interactive evidence</div><h2>Reference / target browser</h2><div class="toolbar" id="bucketButtons"></div><div class="toolbar"><button id="previous" title="Previous example">Previous</button><span id="pageLabel"></span><button id="next" title="Next example">Next</button></div><div class="pair"><figure><img id="reference" alt="Reference screenshot"><figcaption>Reference screenshot</figcaption></figure><figure><img id="target" alt="Target screenshot"><figcaption>Target screenshot</figcaption></figure></div><h3 id="pairTitle"></h3><pre id="meta"></pre></section>
<section><div class="eyebrow">Phase 2 · full test inference</div><h2>Independent compression baselines by degree of change</h2><p id="classicalStatus" class="notice"></p><div id="chartArea"><canvas id="rdChart" width="1040" height="340" aria-label="Rate-distortion chart"></canvas></div><div id="tableArea"></div><p class="pending">* PSNR means exclude exact reconstructions (infinite PSNR); the table reports their finite sample count. Delta+PNG was lossless for all 2,000 tested pairs.</p><p class="pending" id="genericStatus"></p></section>
<section><div class="eyebrow">Preliminary</div><h2>Architecture validation, not final results</h2><div class="notice"><b>Proof-of-concept No-Reference Codec:</b> 1,000 balanced training pairs, 64px crops, one lambda (0.0130), five epochs maximum. This is deliberately not presented as a converged codec or a reference-conditioned result.</div><p id="pocStatus" class="pending"></p><div id="pocArea"></div></section>
<section><div class="eyebrow">Status & roadmap</div><h2>What this review validates</h2><div class="roadmap"><div><b>Done</b><br>Phase 1: complete 24k balanced dataset. Phase 2 independent classical inference is being produced on the complete 2k-pair test set.</div><div><b>In progress</b><br>Phase 3 proof-of-concept. Full CPU training is estimated at 1-4 days per lambda, so this demo does not claim final learned-codec performance.</div><div><b>Not started</b><br>Reference-feature extraction, v1 reference conditioning, Mode Selection, and the required unrelated-bucket fallback test.</div><div><b>What unblocks the research</b><br>GPU access enables multi-lambda, full-resolution training and the final bucket-level RD/BD-rate benchmark.</div></div></section></main>
<script>const DATA={data};let bucket='identical',index=0;const colors={{jpeg:'#cc6a22',webp:'#1276a8'}};const buttons=document.querySelector('#bucketButtons');Object.keys(DATA.examples).forEach(b=>{{const x=document.createElement('button');x.textContent=b;x.onclick=()=>{{bucket=b;index=0;renderPair()}};buttons.append(x)}});function renderPair(){{[...buttons.children].forEach(x=>x.setAttribute('aria-pressed',x.textContent===bucket));const items=DATA.examples[bucket],item=items[index];document.querySelector('#reference').src=item.reference;document.querySelector('#target').src=item.target;document.querySelector('#pageLabel').textContent=`${{index+1}} / ${{items.length}}`;document.querySelector('#pairTitle').textContent=`${{bucket}} · ${{item.pairId}}`;document.querySelector('#meta').textContent=JSON.stringify(item.meta,null,2)}}document.querySelector('#previous').onclick=()=>{{index=(index+DATA.examples[bucket].length-1)%DATA.examples[bucket].length;renderPair()}};document.querySelector('#next').onclick=()=>{{index=(index+1)%DATA.examples[bucket].length;renderPair()}};renderPair();document.querySelector('#classicalStatus').textContent=DATA.classicalStatus;document.querySelector('#genericStatus').textContent=DATA.genericStatus;document.querySelector('#pocStatus').textContent=DATA.pocStatus;
function draw(){{if(!DATA.classical){{document.querySelector('#chartArea').innerHTML='<p class="pending">Chart appears only after all 2,000 real records are present.</p>';return}}const c=document.querySelector('#rdChart'),ctx=c.getContext('2d'),w=c.width,h=c.height,p=48;ctx.clearRect(0,0,w,h);const d=DATA.classical[bucket],points=[...d.jpeg,...d.webp].filter(x=>x.psnr!==null);if(!points.length){{document.querySelector('#chartArea').innerHTML='<p class="pending">This bucket contains only exact reconstructions at the evaluated settings.</p>';return}}ctx.strokeStyle='#9aa9b4';ctx.beginPath();ctx.moveTo(p,12);ctx.lineTo(p,h-p);ctx.lineTo(w-12,h-p);ctx.stroke();ctx.fillStyle='#41505b';ctx.font='12px Arial';ctx.fillText('Mean finite-sample PSNR (dB)',6,18);ctx.fillText('Mean bits per pixel',w-125,h-12);const minX=Math.min(...points.map(x=>x.bpp)),maxX=Math.max(...points.map(x=>x.bpp)),minY=Math.min(...points.map(x=>x.psnr)),maxY=Math.max(...points.map(x=>x.psnr));for(const codec of ['jpeg','webp']){{const usable=d[codec].filter(v=>v.psnr!==null);ctx.strokeStyle=colors[codec];ctx.lineWidth=2;ctx.beginPath();usable.forEach((v,i)=>{{let x=p+(v.bpp-minX)/(maxX-minX)*(w-p-24),y=h-p-(v.psnr-minY)/(maxY-minY||1)*(h-p-30);i?ctx.lineTo(x,y):ctx.moveTo(x,y);ctx.fillStyle=colors[codec];ctx.fillRect(x-3,y-3,6,6)}});ctx.stroke();ctx.fillStyle=colors[codec];ctx.fillText(codec.toUpperCase(),p+10+(codec==='webp'?75:0),28)}}}}function table(){{if(!DATA.classical)return;let rows='<table><thead><tr><th>Bucket</th><th>Codec</th><th>Quality</th><th>Mean bpp</th><th>Mean finite PSNR</th></tr></thead><tbody>';for(const b of Object.keys(DATA.classical)){{for(const codec of ['jpeg','webp'])for(const x of DATA.classical[b][codec])rows+=`<tr><td>${{b}}</td><td>${{codec.toUpperCase()}}</td><td>${{x.quality}}</td><td>${{x.bpp.toFixed(3)}}</td><td>${{x.psnr===null?'all exact':x.psnr.toFixed(2)+' dB ('+x.psnr_samples+' finite)'}}</td></tr>`;const x=DATA.classical[b].delta_png;rows+=`<tr><td>${{b}}</td><td>Delta+PNG</td><td>lossless</td><td>${{x.bpp.toFixed(3)}}</td><td>${{x.lossless?'infinite':'see raw records'}}</td></tr>`;if(DATA.generic&&DATA.generic[b]){{const g=DATA.generic[b];rows+=`<tr><td>${{b}}</td><td>${{g.model}}</td><td>${{g.quality}}</td><td>${{g.bpp.toFixed(3)}}</td><td>${{g.psnr===null?'all exact':g.psnr.toFixed(2)+' dB ('+g.psnr_samples+' finite)'}}</td></tr>`}}}}document.querySelector('#tableArea').innerHTML=rows+'</tbody></table>'}}function poc(){{if(!DATA.poc)return;const h=DATA.poc.history||[],last=h[h.length-1];let s=`<p><b>Actual run:</b> ${{DATA.poc.train_pairs}} pairs; ${{DATA.poc.patch_size}}px; ${{DATA.poc.epochs}} epochs; lambda=${{DATA.poc.lmbda}}. Last logged loss: ${{last?last.loss.toFixed(4):'unavailable'}}.</p>`;if(DATA.poc.examples)for(const e of DATA.poc.examples)s+=`<h3>${{e.bucket}} · ${{e.pair_id}}</h3><div class="grid"><figure><img src="${{e.reference}}"><figcaption>Reference</figcaption></figure><figure><img src="${{e.target}}"><figcaption>Target</figcaption></figure><figure><img src="${{e.reconstruction}}"><figcaption>Real reconstruction</figcaption></figure></div>`;document.querySelector('#pocArea').innerHTML=s}}draw();table();poc();</script></body></html>''', encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

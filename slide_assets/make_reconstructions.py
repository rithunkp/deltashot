"""Assemble an image panel from existing evaluation decodes and test-pair inputs."""

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_pairs_full" / "test"
BUCKETS = ("identical", "minor", "moderate", "major", "unrelated")


def load_font(size, bold=False):
    candidates = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def pair_files(pair_id):
    direct = DATA / pair_id
    for folder in (direct, *direct.glob("*")) if direct.exists() else ():
        ref, target = folder / "reference.png", folder / "target.png"
        if ref.is_file() and target.is_file():
            return ref, target
    for meta_path in DATA.rglob("meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("pair_id") == pair_id or meta_path.parent.name == pair_id:
            ref, target = meta_path.parent / "reference.png", meta_path.parent / "target.png"
            if ref.is_file() and target.is_file():
                return ref, target
    raise FileNotFoundError(f"Could not find reference.png and target.png for {pair_id} under {DATA}")


def paste_image(canvas, draw, path, x, y, width, height):
    with Image.open(path) as source:
        source = source.convert("RGB")
        image = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    ix, iy = x + (width-image.width)//2, y + (height-image.height)//2
    canvas.paste(image, (ix, iy))
    return path.name


def main():
    records = {x["pair_id"]: x for x in (json.loads(line) for line in (RESULTS / "pipeline_eval_modes.jsonl").read_text(encoding="utf-8").splitlines())}
    with (OUT / "reconstruction_sources.csv").open(encoding="utf-8-sig", newline="") as handle:
        selections = list(csv.DictReader(handle))
    if [x["bucket"] for x in selections] != list(BUCKETS):
        raise ValueError("reconstruction_sources.csv must have one row per bucket in project order")

    images = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(images)
    draw.text((34, 14), "Reconstructions by Degree of Change", fill="#142330", font=load_font(30, True))
    draw.text((35, 52), "Evaluation decodes from completed 28-epoch run; Mode Selection reuses the selected codec's decode.", fill="#445663", font=load_font(15))
    draw.text((35, 75), "Reference/target inputs: local test-pair PNGs. Each cell's source path is listed in reconstruction_sources.csv.", fill="#445663", font=load_font(15))
    labels = ("Reference", "Target", "No-Ref decode", "Ref-Cond decode", "Mode Sel choice")
    left, top, col_w = 28, 108, 308
    row_h = 150
    for col, label in enumerate(labels):
        x = left + col * col_w
        draw.rectangle((x, top, x + col_w - 8, top + 31), fill="#eaf0f4")
        draw.text((x + (col_w-8)//2, top+16), label, anchor="mm", fill="#142330", font=load_font(15, True))

    source_rows = []
    for ri, selected in enumerate(selections):
        bucket, pair_id = selected["bucket"], selected["pair_id"]
        if pair_id not in records:
            raise KeyError(f"{pair_id} is absent from pipeline_eval_modes.jsonl")
        record = records[pair_id]
        if record["degree_of_change"] != bucket:
            raise ValueError(f"bucket mismatch for {pair_id}")
        reference, target = pair_files(pair_id)
        noref = ROOT / record["noref_reconstruction"]
        refcond = ROOT / record["reference_reconstruction"]
        if not noref.is_file() or not refcond.is_file():
            raise FileNotFoundError(f"cached evaluation reconstruction missing for {pair_id}: {noref}, {refcond}")
        mode_codec = record["mode_selection"]["codec"]
        mode_img = noref if mode_codec == "noref" else refcond
        if mode_codec not in ("noref", "reference_conditioned"):
            raise ValueError(f"unknown selected mode {mode_codec} for {pair_id}")
        y = top + 37 + ri * row_h
        draw.text((left, y), f"{bucket} · {pair_id} · selected {mode_codec}", fill="#233746", font=load_font(14, True))
        paths = (reference, target, noref, refcond, mode_img)
        for col, source_path in enumerate(paths):
            x = left + col * col_w
            box_y = y + 22
            draw.rectangle((x, box_y, x+col_w-8, box_y+118), fill="#f6f8f9", outline="#d0d9df", width=1)
            paste_image(images, draw, source_path, x+5, box_y+4, col_w-18, 88)
            short = source_path.name
            if col == 4:
                short = f"{mode_codec}: {short}"
            draw.text((x+6, box_y+96), short[:43], fill="#445663", font=load_font(11))
        source_rows.append({"bucket":bucket,"pair_id":pair_id,"mode_selected_codec":mode_codec,
                            "reference_source":str(reference.relative_to(ROOT)),"target_source":str(target.relative_to(ROOT)),
                            "noref_decode_source":str(noref.relative_to(ROOT)),"refcond_decode_source":str(refcond.relative_to(ROOT)),
                            "mode_selection_image_source":str(mode_img.relative_to(ROOT)),
                            "note":"Mode Selection output uses the corresponding evaluation decoder image; no new decode was run."})
    with (OUT / "reconstruction_sources.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(source_rows[0]))
        writer.writeheader();writer.writerows(source_rows)
    images.save(OUT / "reconstructions.png")
    with Image.open(OUT / "reconstructions.png") as check:
        check.verify()
    print(f"Wrote {OUT / 'reconstructions.png'} from five test pairs and existing evaluation decodes.")


if __name__ == "__main__":
    main()

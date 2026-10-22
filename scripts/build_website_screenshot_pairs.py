"""Prepare provenance-labeled website screenshot pairs for the local demo."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageStat

from data.bucketing import BUCKETS, bucket_from_ratio, compute_changed_pixel_ratio


ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "website_screenshots"
TEST_ROOT = ROOT / "data" / "synthetic_pairs_full" / "test"
BUCKET_PAIRS = (
    ("python_tutorial_311_312", "Python tutorial", "https://docs.python.org/3.11/tutorial/index.html",
     "https://docs.python.org/3.12/tutorial/index.html"),
    ("python_asyncio_311_312", "Python asyncio tasks", "https://docs.python.org/3.11/library/asyncio-task.html",
     "https://docs.python.org/3.12/library/asyncio-task.html"),
    ("python_typing_311_312", "Python typing reference", "https://docs.python.org/3.11/library/typing.html",
     "https://docs.python.org/3.12/library/typing.html"),
    ("bootstrap_accordion_52_53", "Bootstrap accordion documentation",
     "https://getbootstrap.com/docs/5.2/components/accordion/",
     "https://getbootstrap.com/docs/5.3/components/accordion/"),
    ("bootstrap_navbar_52_53", "Bootstrap navbar documentation",
     "https://getbootstrap.com/docs/5.2/components/navbar/",
     "https://getbootstrap.com/docs/5.3/components/navbar/"),
    ("bootstrap_cards_52_53", "Bootstrap card documentation",
     "https://getbootstrap.com/docs/5.2/components/card/",
     "https://getbootstrap.com/docs/5.3/components/card/"),
)
ERROR_MARKERS = ("404 not found", "page not found", "access denied", "just a moment",
                 "verify you are human", "this site can’t be reached", "this site can't be reached")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_error(path: Path, title: str, body_text: str) -> str | None:
    if not title.strip() or any(marker in title.lower() for marker in ERROR_MARKERS):
        return f"page title indicates an error or is empty: {title!r}"
    if any(marker in body_text.lower() for marker in ERROR_MARKERS):
        return "page content contains a common error/interstitial marker"
    if len(body_text.strip()) < 120:
        return f"page has too little visible text ({len(body_text.strip())} characters)"
    with Image.open(path) as image:
        if image.size != (512, 512):
            return f"capture dimensions are {image.size}, expected (512, 512)"
        if ImageStat.Stat(image.convert("RGB")).stddev == [0.0, 0.0, 0.0]:
            return "capture is a flat/blank image"
    return None


async def capture_version_pairs(dropped: list[dict]) -> list[dict]:
    from playwright.async_api import async_playwright

    records = []
    async with async_playwright() as playwright:
        candidates = [os.environ.get("CHROME_PATH"), shutil.which("google-chrome"),
                      shutil.which("chrome"), shutil.which("chrome.exe"),
                      r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
        chrome = next((item for item in candidates if item and Path(item).is_file()), None)
        launch_options = {"headless": True, "timeout": 45_000,
                          "args": ["--disable-crash-reporter", "--disable-breakpad",
                                   "--disable-crashpad-for-testing", "--disable-gpu"]}
        if chrome:
            launch_options["executable_path"] = chrome
        browser = await playwright.chromium.launch(**launch_options)
        try:
            context = await browser.new_context(viewport={"width": 512, "height": 512}, device_scale_factor=1)
            for pair_id, description, reference_url, target_url in BUCKET_PAIRS:
                pair_dir = GALLERY / "real_versions" / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)
                captured = []
                failure = None
                for role, url in (("reference", reference_url), ("target", target_url)):
                    page = await context.new_page()
                    try:
                        response = await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                        if response is None or response.status >= 400:
                            raise RuntimeError(f"HTTP status {response.status if response else 'unknown'}")
                        try:
                            await page.wait_for_load_state("networkidle", timeout=10_000)
                        except Exception:
                            pass
                        await page.evaluate("document.fonts.ready")
                        await page.add_style_tag(content="""
                          #wm-ipp, #wm-ipp-base, .wayback-toolbar,
                          #onetrust-banner-sdk, .cookie-banner, [aria-label*='cookie' i] { display:none!important }
                          *,*::before,*::after { animation:none!important; transition:none!important; caret-color:transparent!important }
                        """)
                        await page.wait_for_timeout(300)
                        screenshot = pair_dir / f"{role}.png"
                        await page.screenshot(path=str(screenshot), full_page=False, animations="disabled")
                        title = await page.title()
                        body = await page.locator("body").inner_text(timeout=5_000)
                        issue = _quality_error(screenshot, title, body)
                        if issue:
                            raise RuntimeError(issue)
                        captured.append({"role": role, "path": screenshot, "url": url, "title": title})
                    except Exception as error:
                        failure = f"{role} capture failed: {error}"
                        break
                    finally:
                        await page.close()
                if failure or len(captured) != 2:
                    dropped.append({"id": pair_id, "origin": "real", "reason": failure or "incomplete pair"})
                    for item in captured:
                        item["path"].unlink(missing_ok=True)
                    continue
                if _sha(captured[0]["path"]) == _sha(captured[1]["path"]):
                    dropped.append({"id": pair_id, "origin": "real", "reason": "reference and target captures are byte-identical"})
                    for item in captured:
                        item["path"].unlink(missing_ok=True)
                    continue
                ratio = compute_changed_pixel_ratio(captured[0]["path"], captured[1]["path"])
                records.append({
                    "id": pair_id,
                    "reference": captured[0]["path"].relative_to(GALLERY).as_posix(),
                    "target": captured[1]["path"].relative_to(GALLERY).as_posix(),
                    "origin": "real",
                    "source_url": {"reference": reference_url, "target": target_url},
                    "capture_date_utc": datetime.now(timezone.utc).isoformat(),
                    "reference_title": captured[0]["title"],
                    "target_title": captured[1]["title"],
                    "changed_pixel_ratio": ratio,
                    "bucket": bucket_from_ratio(ratio),
                    "description": description + " across documented major versions.",
                })
                print(f"captured {pair_id}: {ratio:.3%} changed ({records[-1]['bucket']})", flush=True)
            await context.close()
        finally:
            await browser.close()
    return records


def copy_synthetic_pairs() -> list[dict]:
    groups = {bucket: [] for bucket in BUCKETS}
    for pair_dir in sorted(TEST_ROOT.glob("pair_*")):
        meta_path = pair_dir / "meta.json"
        if not (meta_path.is_file() and (pair_dir / "reference.png").is_file()
                and (pair_dir / "target.png").is_file()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bucket = meta.get("degree_of_change")
        if bucket in groups:
            groups[bucket].append((pair_dir, meta))
    records = []
    for bucket, pairs in groups.items():
        if len(pairs) < 2:
            raise RuntimeError(f"need at least two test pairs for {bucket}, found {len(pairs)}")
        chosen = [pairs[(len(pairs) - 1) // 3], pairs[2 * (len(pairs) - 1) // 3]]
        for index, (source, meta) in enumerate(chosen, 1):
            pair_id = f"sample_{bucket}_{index}"
            destination = GALLERY / "benchmark_samples" / pair_id
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "reference.png", destination / "reference.png")
            shutil.copy2(source / "target.png", destination / "target.png")
            shutil.copy2(source / "meta.json", destination / "meta.json")
            records.append({
                "id": pair_id,
                "reference": (destination / "reference.png").relative_to(GALLERY).as_posix(),
                "target": (destination / "target.png").relative_to(GALLERY).as_posix(),
                "origin": "synthetic",
                "source_url": None,
                "capture_date_utc": None,
                "bucket": bucket,
                "changed_pixel_ratio": meta.get("changed_pixel_ratio"),
                "description": f"Selected from synthetic test split as a visual {bucket} example; demo only, not a new evaluation.",
            })
    return records


def copy_unrelated_pairs() -> list[dict]:
    source_manifest = GALLERY / "capture_manifest.json"
    source_data = json.loads(source_manifest.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in source_data.get("captures", []) if item.get("status") == "captured"}
    wanted = (("unrelated_github_wikipedia", "github_explore", "wikipedia_home"),
              ("unrelated_mdn_python", "mdn_css_grid", "python_tutorial"))
    records = []
    for pair_id, ref_name, target_name in wanted:
        if ref_name not in by_name or target_name not in by_name:
            raise RuntimeError(f"existing capture manifest is missing {ref_name} or {target_name}")
        ref_record, target_record = by_name[ref_name], by_name[target_name]
        pair_dir = GALLERY / "unrelated_pairs" / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        reference_source = GALLERY / ref_record["screenshot"]
        target_source = GALLERY / target_record["screenshot"]
        shutil.copy2(reference_source, pair_dir / "reference.png")
        shutil.copy2(target_source, pair_dir / "target.png")
        ratio = compute_changed_pixel_ratio(pair_dir / "reference.png", pair_dir / "target.png")
        records.append({
            "id": pair_id,
            "reference": (pair_dir / "reference.png").relative_to(GALLERY).as_posix(),
            "target": (pair_dir / "target.png").relative_to(GALLERY).as_posix(),
            "origin": "real",
            "source_url": {"reference": ref_record["requested_url"], "target": target_record["requested_url"]},
            "capture_date_utc": source_data.get("captured_at_utc"),
            "bucket": "unrelated",
            "changed_pixel_ratio": ratio,
            "description": f"Independent public pages ({ref_record.get('title')} vs {target_record.get('title')}); unrelated fallback demonstration only.",
        })
    return records


def build_manifest(capture: bool) -> dict:
    for name in ("real_versions", "benchmark_samples", "unrelated_pairs"):
        (GALLERY / name).mkdir(parents=True, exist_ok=True)
    dropped = []
    real_records = asyncio.run(capture_version_pairs(dropped)) if capture else []
    records = real_records + copy_synthetic_pairs() + copy_unrelated_pairs()
    data = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local live/premade demo material only.",
        "benchmark_note": "Real website captures and copied synthetic examples in this gallery are not a new benchmark or training data. Synthetic examples retain their source split and provenance.",
        "capture_settings": {"viewport": [512, 512], "device_scale_factor": 1, "full_page": False,
                             "browser": "Playwright Chromium", "same_browser_context_per_pair": True},
        "bucket_rule": "data.bucketing.bucket_from_ratio; identical <=0.01%, minor <2%, moderate <15%, major <=50%, unrelated >50% or explicitly unrelated.",
        "pairs": records,
        "dropped": dropped,
    }
    destination = GALLERY / "manifest.json"
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} pairs ({len(real_records)} captured real-version pairs); dropped={len(dropped)}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-capture", action="store_true", help="reuse existing real_versions images")
    args = parser.parse_args()
    build_manifest(capture=not args.skip_capture)


if __name__ == "__main__":
    main()

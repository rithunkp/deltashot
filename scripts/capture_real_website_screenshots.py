"""Capture public website pages as standalone, provenance-documented screenshots."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from html.parser import HTMLParser

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "real_website_screenshots"
SITES = [
    ("wikipedia_home", "Wikipedia home page", "https://www.wikipedia.org/"),
    ("github_explore", "GitHub Explore", "https://github.com/explore"),
    ("python_tutorial", "Python tutorial", "https://docs.python.org/3/tutorial/index.html"),
    ("mdn_css_grid", "MDN CSS Grid guide", "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout"),
]


class TitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        self.in_title = tag.lower() == "title"

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.parts.append(data.strip())


def find_chrome(explicit_path: str | None) -> str:
    candidates = [explicit_path] if explicit_path else []
    candidates.extend((
        os.environ.get("CHROME_PATH"),
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "Chrome executable not found; pass --chrome or set CHROME_PATH."
    )


def capture(chrome: str, width: int, height: int, timeout_ms: int) -> list[dict]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    captures = []
    profile = OUTPUT / ".chrome-profile"
    profile.mkdir(parents=True, exist_ok=True)
    try:
        for index, (name, label, url) in enumerate(SITES):
            destination = OUTPUT / f"{name}.png"
            record = {"name": name, "requested_url": url}
            command = [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                "--disable-crash-reporter",
                "--disable-breakpad",
                "--disable-crashpad-for-testing",
                "--disable-features=Crashpad",
                f"--user-data-dir={profile / str(index)}",
                f"--window-size={width},{height}",
                "--force-device-scale-factor=1",
                "--run-all-compositor-stages-before-draw",
                "--timeout=4000",
                f"--screenshot={destination}",
                "--dump-dom",
                url,
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=(timeout_ms / 1000) + 15,
                    check=False,
                )
                if not destination.is_file() or destination.stat().st_size == 0:
                    raise RuntimeError(
                        completed.stderr[-2000:] or f"Chrome exited {completed.returncode}"
                    )
                with Image.open(destination) as screenshot:
                    screenshot.verify()
                title_parser = TitleParser()
                title_parser.feed(completed.stdout)
                page_title = " ".join(title_parser.parts)
                if "\ufffd" in page_title or not page_title:
                    page_title = label
                record.update({
                    "title": page_title,
                    "screenshot": destination.name,
                    "width": width,
                    "height": height,
                    "bytes": destination.stat().st_size,
                    "status": "captured",
                })
                if completed.returncode != 0:
                    record["chrome_exit_code"] = completed.returncode
                    record["chrome_warning"] = completed.stderr[-2000:]
                print(f"captured {destination.name}: {record['title']} ({record['bytes']} bytes)", flush=True)
            except Exception as error:
                destination.unlink(missing_ok=True)
                record.update({"status": "failed", "error": str(error)})
                print(f"failed {name}: {error}", flush=True)
            captures.append(record)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "viewport": {"width": width, "height": height, "device_scale_factor": 1},
        "purpose": "Standalone real website screenshots for interactive codec demonstrations.",
        "pairing_note": "These are independently captured pages, not matched before/after version pairs and not benchmark data.",
        "captures": captures,
    }
    (OUTPUT / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return captures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", help="Chrome executable path; defaults to installed Chrome")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()
    if args.width < 320 or args.height < 320:
        parser.error("viewport dimensions must both be at least 320 pixels")
    captures = capture(find_chrome(args.chrome), args.width, args.height, args.timeout_ms)
    if not any(item["status"] == "captured" for item in captures):
        raise SystemExit("No sites could be captured; inspect the errors above.")
    print(f"\nSaved {sum(item['status'] == 'captured' for item in captures)} screenshots in {OUTPUT}")


if __name__ == "__main__":
    main()

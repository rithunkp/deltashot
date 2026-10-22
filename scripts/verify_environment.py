from __future__ import annotations

import argparse
import asyncio
import importlib
from pathlib import Path

from data.rendering import render_html_to_png


def verify_import(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "available")


async def verify_playwright(output_path: Path) -> None:
    html = "<!doctype html><html><body style='margin:0;background:#fff'><h1>DeltaShot</h1></body></html>"
    await render_html_to_png(html, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DeltaShot Phase 0 dependencies.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_pairs/playwright_check.png"))
    args = parser.parse_args()

    versions = {
        "compressai": verify_import("compressai"),
        "playwright": verify_import("playwright"),
        "PIL": verify_import("PIL"),
        "bs4": verify_import("bs4"),
    }
    asyncio.run(verify_playwright(args.output))
    for name, version in versions.items():
        print(f"{name}: {version}")
    print(f"playwright_render: wrote {args.output}")


if __name__ == "__main__":
    main()

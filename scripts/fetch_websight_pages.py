"""Cache WebSight v0.2 HTML records through the public Dataset Viewer API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


API_URL = "https://datasets-server.huggingface.co/rows"


def fetch_rows(offset: int, length: int) -> dict:
    query = urlencode(
        {
            "dataset": "HuggingFaceM4/WebSight",
            "config": "v0.2",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    with urlopen(f"{API_URL}?{query}", timeout=60) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache WebSight v0.2 HTML records for DeltaShot.")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/raw_websight/websight_v02_2000.json"))
    args = parser.parse_args()

    records = []
    for offset in range(args.offset, args.offset + args.count, 100):
        payload = fetch_rows(offset, min(100, args.offset + args.count - offset))
        for item in payload["rows"]:
            html = item["row"].get("text", "")
            if "<html" not in html.lower():
                continue
            records.append(
                {
                    "page_id": f"websight_v02_{item['row_idx']:07d}",
                    "html": html,
                    "source": "HuggingFaceM4/WebSight:v0.2",
                    "idea": item["row"].get("llm_generated_idea", ""),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding="utf-8")
    print(json.dumps({"cached": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

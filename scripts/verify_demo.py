"""End-to-end local demo verification; --stub-codecs exercises only the web layer."""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REAL_SCREENSHOTS = ROOT / "data" / "real_website_screenshots"
LEGACY_SCREENSHOTS = ROOT / "website_screenshots"
TEST_ROOT = ROOT / "data" / "synthetic_pairs_full" / "test"
REPORT_JSON = ROOT / "results" / "demo_verify_report.json"
REPORT_TEXT = ROOT / "results" / "demo_verify_report.txt"
PAGE_LIVE = ROOT / "results" / "demo_verify_live.png"
PAGE_PREMADE = ROOT / "results" / "demo_verify_premade.png"


def _import_demo(stub_codecs):
    if stub_codecs:
        fake_package = ModuleType("compressai")
        fake_models = ModuleType("compressai.models")
        fake_models.ScaleHyperprior = object
        fake_package.models = fake_models
        sys.modules.setdefault("compressai", fake_package)
        sys.modules.setdefault("compressai.models", fake_models)
    from scripts import local_demo
    return local_demo


class StubCodecRunner:
    """Deterministic image-file codecs for exercising HTTP/UI wiring without CompressAI."""

    def __init__(self, demo):
        self.demo = demo

    def run(self, target_image, target_upload_bytes, reference_image=None):
        demo = self.demo
        run_id = uuid.uuid4().hex[:12]
        out_dir = demo.OUTPUT_ROOT / run_id
        out_dir.mkdir(parents=True)
        uploaded_size = target_image.size
        target = demo.prepare_working_image(target_image)
        working_size = target.size
        reference = demo.prepare_working_image(reference_image) if reference_image else None
        results = [demo.CodecRunner._result(
            run_id, "Original target (working image)", target, target_upload_bytes,
            note="Stub verification input; metrics use the 512x512 working image.",
            psnr=math.inf, original_size=working_size,
        )]

        def save_result(label, image, filename, data, *, lossless=False, note=""):
            (out_dir / filename).write_bytes(data)
            decoded = Image.open(io.BytesIO(data)).convert("RGB")
            if lossless:
                decoded = target.copy()
                buffer = io.BytesIO()
                decoded.save(buffer, format="PNG")
                data = buffer.getvalue()
                (out_dir / filename).write_bytes(data)
            result = demo.CodecRunner._result(
                run_id, label, decoded, len(data), filename=filename,
                psnr=demo.psnr_db(target, decoded), note=note, lossless=lossless,
                original_size=working_size,
            )
            results.append(result)
            return result, data, decoded

        jpeg_buffer = io.BytesIO()
        target.save(jpeg_buffer, format="JPEG", quality=75)
        save_result("JPEG (quality 75) [stub]", target, "target_q75.jpg", jpeg_buffer.getvalue())
        webp_buffer = io.BytesIO()
        target.save(webp_buffer, format="WEBP", quality=75)
        save_result("WebP (quality 75) [stub]", target, "target_q75.webp", webp_buffer.getvalue())

        if reference is not None:
            delta_buffer = io.BytesIO()
            target.save(delta_buffer, format="PNG")
            save_result("Delta+PNG (lossless) [stub]", target, "delta_png.bin",
                        delta_buffer.getvalue(), lossless=True,
                        note="Stubbed lossless payload for web-layer verification.")
        else:
            results.append(demo.CodecRunner._unavailable(
                "Delta+PNG (lossless)", "Upload a reference to enable this path."))

        noref_buffer = io.BytesIO()
        target.save(noref_buffer, format="JPEG", quality=65)
        noref_item, noref_data, noref_image = save_result(
            "No-Reference neural (stub)", target, "noref.bin", noref_buffer.getvalue(),
            note="Stub codec: not a CompressAI bitstream.")

        ref_item = None
        ref_data = None
        ref_image = None
        if reference is not None:
            ref_buffer = io.BytesIO()
            target.save(ref_buffer, format="JPEG", quality=70)
            ref_item, ref_data, ref_image = save_result(
                "Reference-Conditioned neural (stub)", target, "reference_conditioned.bin",
                ref_buffer.getvalue(), note="Stub codec: not a CompressAI bitstream.")
        else:
            results.append(demo.CodecRunner._unavailable(
                "Reference-Conditioned neural (lossy)",
                "Skipped: upload a reference to enable this codec."))

        mode_manifest = {"status": "skipped_no_reference", "codec": None,
                         "reason": "Mode Selection skipped because no reference was supplied."}
        if ref_item is not None:
            selected = noref_item
            selected_data = noref_data
            selected_image = Image.open(io.BytesIO(noref_data)).convert("RGB")
            selected, _, _ = save_result(
                "Mode Selection → No-Reference [stub]", selected_image, "mode_selected.bin",
                selected_data, note="Stub-only mode wrapper.")
            mode_manifest = {"status": "selected", "codec": "noref",
                             "selected_file": "mode_selected.bin", "candidate_objectives": {}}
        else:
            results.append(demo.CodecRunner._unavailable("Mode Selection", mode_manifest["reason"]))

        manifest = {
            "run_id": run_id,
            "original_size": list(uploaded_size),
            "working_size": list(working_size),
            "reference_provided": reference is not None,
            "mode_selection": mode_manifest,
            "neural_decode_source": "stub decoders reopened compressed files from disk",
            "neural_bitstreams": {
                "noref": {"file": "noref.bin", "decoded_from_disk": True},
                "reference_conditioned": ({"file": "reference_conditioned.bin", "decoded_from_disk": True}
                                           if ref_item is not None else None),
            },
            "stub_codecs": True,
            "files": {path.name: path.stat().st_size for path in out_dir.iterdir() if path.is_file()},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["files"]["manifest.json"] = (out_dir / "manifest.json").stat().st_size
        return {"run_id": run_id, "results": results, "manifest": manifest}


def _fixture_screenshots(temp_dir):
    pairs = sorted(path for path in TEST_ROOT.glob("pair_*") if (path / "target.png").is_file())
    if len(pairs) < 4:
        raise FileNotFoundError(f"need four test-pair images to make stub fixtures under {TEST_ROOT}")
    output = []
    for index, pair in enumerate(pairs[:4]):
        path = Path(temp_dir) / f"fixture_{index + 1}.png"
        Image.open(pair / "target.png").convert("RGB").save(path)
        output.append(path)
    return output


def _prepare_stub_premades(demo, pairs, temp_dir):
    demo.PREMADE_ROOT = Path(temp_dir) / "premade"
    demo.PREMADE_RUNS = demo.PREMADE_ROOT / "runs"
    demo.OUTPUT_ROOT = demo.PREMADE_RUNS
    demo.PREMADE_RUNS.mkdir(parents=True, exist_ok=True)
    runner = StubCodecRunner(demo)
    records = []
    for pair in pairs:
        ref_path = ROOT / "website_screenshots" / pair["reference"]
        target_path = ROOT / "website_screenshots" / pair["target"]
        with Image.open(ref_path) as ref, Image.open(target_path) as target:
            result = runner.run(target.convert("RGB"), target_path.stat().st_size, ref.convert("RGB"))
        records.append({**pair, "run_id": result["run_id"],
                        "working_size": result["manifest"]["working_size"],
                        "mode_selection": result["manifest"]["mode_selection"],
                        "neural_decode_source": result["manifest"]["neural_decode_source"],
                        "methods": [{key: value for key, value in item.items() if key != "image_data"}
                                    for item in result["results"]]})
    demo.PREMADE_ROOT.mkdir(parents=True, exist_ok=True)
    (demo.PREMADE_ROOT / "manifest.json").write_text(
        json.dumps({"pairs": records, "codec_mode": "STUB ONLY"}, indent=2), encoding="utf-8")


def _request(url, payload=None, timeout=1800):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except urllib.error.URLError as error:
        return 0, str(error).encode("utf-8")


def _upload(path):
    return {"filename": path.name, "data": base64.b64encode(path.read_bytes()).decode("ascii")}


def _image_from_response(raw):
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image


def _console_text(value):
    return str(value).replace("∞ dB (exact)", "exact (lossless)").encode("ascii", "replace").decode("ascii")


def verify_response(base_url, label, target_path, reference_path, report, expect_reference):
    payload = {"target": _upload(target_path),
               "reference": _upload(reference_path) if reference_path else None}
    status, body = _request(base_url + "/api/compress", payload)
    if status != 200:
        raise AssertionError(f"{label}: HTTP {status}: {body.decode('utf-8', 'replace')}")
    data = json.loads(body)
    manifest = data.get("manifest")
    if not manifest or not manifest.get("run_id"):
        raise AssertionError(f"{label}: response is missing its run manifest")
    status_manifest, manifest_bytes = _request(base_url + f"/files/{manifest['run_id']}/manifest.json")
    if status_manifest != 200 or json.loads(manifest_bytes).get("run_id") != manifest["run_id"]:
        raise AssertionError(f"{label}: persisted manifest is missing or invalid")
    if manifest.get("working_size") != [512, 512]:
        raise AssertionError(f"{label}: working_size is {manifest.get('working_size')}, expected [512, 512]")
    if manifest.get("original_size") != list(Image.open(target_path).size):
        raise AssertionError(f"{label}: original upload dimensions were not preserved in the manifest")
    if not manifest.get("neural_decode_source") or "disk" not in manifest["neural_decode_source"]:
        raise AssertionError(f"{label}: neural decode is not documented as disk-backed")
    for codec, bitstream in manifest.get("neural_bitstreams", {}).items():
        if bitstream is not None and not bitstream.get("decoded_from_disk"):
            raise AssertionError(f"{label}: {codec} decode was not recorded as disk-backed")
    if manifest.get("mode_selection", {}).get("status") not in {"selected", "skipped_no_reference"}:
        raise AssertionError(f"{label}: Mode Selection status is not recorded")
    if expect_reference and manifest["mode_selection"].get("status") != "selected":
        raise AssertionError(f"{label}: reference case should record a Mode Selection choice")
    if not expect_reference and manifest["mode_selection"].get("status") != "skipped_no_reference":
        raise AssertionError(f"{label}: no-reference case should explicitly skip Mode Selection")

    run_id = manifest["run_id"]
    records = []
    for item in data.get("results", []):
        if item.get("available") is False:
            continue
        filename = item.get("filename")
        if filename:
            status_file, file_data = _request(base_url + f"/files/{run_id}/{filename}")
            if status_file != 200 or len(file_data) <= 0:
                raise AssertionError(f"{label}/{filename}: compressed artifact missing or empty")
            if item.get("bytes") != len(file_data):
                raise AssertionError(f"{label}/{filename}: reported bytes {item.get('bytes')} != file size {len(file_data)}")
            expected_manifest_size = manifest.get("files", {}).get(filename)
            if expected_manifest_size is not None and expected_manifest_size != len(file_data):
                raise AssertionError(f"{label}/{filename}: manifest byte count mismatch")
        decoded_url = item.get("decoded_file_url")
        if decoded_url:
            status_png, png_data = _request(base_url + decoded_url)
            if status_png != 200:
                raise AssertionError(f"{label}/{filename}: decoded PNG is missing")
            decoded = _image_from_response(png_data)
            if decoded.size != (512, 512):
                raise AssertionError(f"{label}/{filename}: decoded image size is {decoded.size}")
        if filename and not item.get("lossless"):
            if item.get("psnr_db") is None:
                raise AssertionError(f"{label}/{filename}: lossy result has no numeric PSNR")
            psnr = float(item["psnr_db"])
            if not math.isfinite(psnr) or psnr <= 15:
                raise AssertionError(f"{label}/{filename}: implausible PSNR {psnr}")
        if filename and filename.endswith(".bin"):
            if not item.get("decoded_file_url"):
                raise AssertionError(f"{label}/{filename}: no saved decoded PNG")
        records.append({"method": item.get("label"), "bytes": item.get("bytes"),
                        "psnr_db": item.get("psnr_db"), "psnr_label": item.get("psnr_label")})

    binaries = [name for name in manifest.get("files", {}) if name.endswith(".bin")]
    for filename in binaries:
        status_file, file_data = _request(base_url + f"/files/{run_id}/{filename}")
        if status_file != 200 or not file_data:
            raise AssertionError(f"{label}: .bin artifact {filename} is missing or empty")
        if manifest["files"].get(filename) != len(file_data):
            raise AssertionError(f"{label}: manifest size for {filename} does not match disk bytes")
    if not any(name == "noref.bin" for name in binaries):
        raise AssertionError(f"{label}: No-Reference .bin is missing")
    if expect_reference and not any(name == "reference_conditioned.bin" for name in binaries):
        raise AssertionError(f"{label}: Reference-Conditioned .bin is missing")
    if not expect_reference and any(name == "reference_conditioned.bin" for name in binaries):
        raise AssertionError(f"{label}: reference codec should be skipped without a reference")
    report.append({"check": label, "status": "PASS", "run_id": run_id,
                   "original_size": manifest["original_size"], "working_size": manifest["working_size"],
                   "mode_selection": manifest["mode_selection"], "methods": records})
    print(f"PASS {label}: upload {manifest['original_size']} -> working {manifest['working_size']}")
    for record in records:
        if record["bytes"] is not None:
            print(f"  {_console_text(record['method'])}: {record['bytes']} bytes, PSNR {_console_text(record['psnr_label'])}")


def verify_invalid(base_url, label, payload, expected_error, report):
    status, body = _request(base_url + "/api/compress", payload)
    data = json.loads(body)
    if status != 400 or data.get("error") != expected_error:
        raise AssertionError(f"{label}: expected HTTP 400 error {expected_error!r}, got {status} {data!r}")
    report.append({"check": label, "status": "PASS", "http_status": status, "error": data["error"]})
    print(f"PASS {label}: HTTP {status}, {data['error']}")


def verify_error_contract(base_url, demo, target_path, report, stub_codecs):
    status, body = _request(base_url + "/not-a-route")
    data = json.loads(body)
    if status != 404 or not data.get("error"):
        raise AssertionError(f"unknown route did not return a JSON error: HTTP {status}, {data!r}")
    report.append({"check": "non-200 JSON error contract", "status": "PASS",
                   "http_status": status, "error": data["error"]})
    print(f"PASS non-200 JSON error contract: HTTP {status}, {data['error']}")
    if stub_codecs:
        original_runner = demo.DemoHandler.runner

        class FailingRunner:
            def run(self, *_args, **_kwargs):
                raise RuntimeError("simulated stub codec failure")

        demo.DemoHandler.runner = FailingRunner()
        try:
            status, body = _request(base_url + "/api/compress", {"target": _upload(target_path), "reference": None})
        finally:
            demo.DemoHandler.runner = original_runner
        data = json.loads(body)
        expected = "Compression failed: simulated stub codec failure"
        if status != 500 or data.get("error") != expected:
            raise AssertionError(f"internal error response was not specific JSON: HTTP {status}, {data!r}")
        report.append({"check": "specific HTTP 500 diagnostic", "status": "PASS", "error": data["error"]})
        print(f"PASS specific HTTP 500 diagnostic: {data['error']}")


def verify_browser(base_url, target_path, browser_channel=None):
    from playwright.sync_api import sync_playwright

    console_errors = []
    page_errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel=browser_channel)
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(base_url, wait_until="domcontentloaded", timeout=120_000)
        if page.locator('[role="tab"]').count() != 2:
            raise AssertionError("both demo tabs did not render")
        page.locator("#target").set_input_files(str(target_path))
        if not page.locator("#pair-preview").is_hidden():
            raise AssertionError("direct target upload left a stale pair preview visible")
        page.locator("#run").click()
        page.locator(".codec").first.wait_for(state="visible", timeout=1_200_000)
        page.wait_for_function("document.querySelector('#live-status').textContent.startsWith('Complete.')",
                               timeout=1_200_000)
        visible_text = page.locator("body").inner_text().lower()
        if "request failed" in visible_text or "compression failed" in visible_text or "failed" in visible_text:
            raise AssertionError("browser UI displays a failed state")
        page.locator("#compare-slider").fill("72")
        page.wait_for_function("document.querySelector('#sliderTop').style.clipPath.includes('28%')")
        PAGE_LIVE.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(PAGE_LIVE), full_page=True)
        live_count = page.locator("#results .codec").count()

        page.locator("#tab-premade").click()
        page.locator("#gallery .pair-card").first.wait_for(state="visible", timeout=20_000)
        page.wait_for_function("Array.from(document.querySelectorAll('#gallery img')).every(x=>x.complete&&x.naturalWidth>0)", timeout=20_000)
        page.screenshot(path=str(PAGE_PREMADE), full_page=True)
        gallery_count = page.locator("#gallery .pair-card").count()
        page.locator("#gallery .pair-card").first.click()
        page.wait_for_function("document.querySelector('#live-status').textContent.includes('premade codec outputs')", timeout=20_000)
        page.locator("#tab-premade").click()
        page.locator("#mode-pair option").nth(1).wait_for(state="attached", timeout=20_000)
        page.locator("#mode-pair").select_option(index=1)
        page.locator("#lambda").evaluate("el=>{el.value='0.027';el.dispatchEvent(new Event('input',{bubbles:true}))}")
        if "Selected:" not in page.locator("#mode-result").inner_text():
            raise AssertionError("Mode Selection lambda explorer did not update")
        page.set_viewport_size({"width": 400, "height": 900})
        page.wait_for_timeout(250)
        if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
            raise AssertionError("live/premade page has horizontal overflow at 400px")
        broken = page.locator("img").evaluate_all("items=>items.filter(x=>!x.complete||x.naturalWidth===0).map(x=>x.src)")
        if broken:
            raise AssertionError(f"broken image(s) in demo: {broken[:5]}")
        if console_errors or page_errors:
            raise AssertionError(f"browser console/page errors: {console_errors[:5]} {page_errors[:5]}")

        page.route("**/api/compress", lambda route: route.fulfill(
            status=500, content_type="application/json",
            body=json.dumps({"error": "simulated server diagnostic"}),
        ))
        page.locator("#tab-live").click()
        page.locator("#run").click()
        page.locator("#live-status").get_by_text("simulated server diagnostic").wait_for(timeout=10_000)
        if page.locator("#live-status").inner_text() != "simulated server diagnostic":
            raise AssertionError("UI did not display the server's specific error text")
        browser.close()
    return {"live_cards": live_count, "gallery_pairs": gallery_count,
            "screenshots": [str(PAGE_LIVE), str(PAGE_PREMADE)], "console_errors": 0, "page_errors": 0}


def write_reports(checks, error=None, stub=False):
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAIL" if error else "PASS", "stub_codecs": stub,
              "checks": checks, "error": error}
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"DeltaShot demo verification: {report['status']}",
             f"Codec mode: {'STUB (web-layer only)' if stub else 'CompressAI'}"]
    for item in checks:
        lines.append(f"{item['status']} {item['check']}")
        if item.get("error"):
            lines.append(f"  error: {item['error']}")
        for method in item.get("methods", []):
            lines.append(f"  {method['method']}: bytes={method['bytes']}, PSNR={method['psnr_label']}")
    if error:
        lines += ["FAIL details:", error]
    REPORT_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--stub-codecs", action="store_true",
                        help="use fake codecs for HTTP/UI verification; never validates CompressAI")
    parser.add_argument("--skip-browser", action="store_true",
                        help="skip Playwright only for environments without a local browser")
    parser.add_argument("--browser-channel", choices=("chrome", "msedge"), default=None,
                        help="use an installed Chrome/Edge instead of Playwright's bundled Chromium")
    args = parser.parse_args()
    checks = []
    server = None
    owns_server = False
    error_text = None
    ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)
    fixture_dir = ROOT / "results" / f".demo_verify_fixtures_{uuid.uuid4().hex[:8]}"
    try:
        demo = _import_demo(args.stub_codecs)
        base_url = f"http://{args.host}:{args.port}"
        status, body = _request(base_url + "/health", timeout=2)
        health = json.loads(body) if status == 200 else {}
        compatible_server = (
            health.get("ready")
            and health.get("api_version") == 3
            and (args.stub_codecs or health.get("neural_available") is True)
        )
        if health.get("ready") and not compatible_server:
            mode = "classical-only" if health.get("neural_available") is False else "stale API"
            print(f"Stopping incompatible {mode} demo server before verification.", flush=True)
            _request(base_url + "/api/shutdown", {})
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                status, body = _request(base_url + "/health", timeout=2)
                current = json.loads(body) if status == 200 else {}
                if not current.get("ready"):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("stale demo server did not stop; run scripts/wsl/stop_demo.ps1 and retry")
            status, body = 0, b""
            health = {}
        if status == 200 and compatible_server:
            print(f"Reusing ready demo server at {base_url}")
        else:
            try:
                server = ThreadingHTTPServer((args.host, args.port), demo.DemoHandler)
                owns_server = True
            except OSError as bind_error:
                print(f"Port {args.port} is occupied; waiting for its demo server to become ready.", flush=True)
                deadline = time.monotonic() + args.startup_timeout
                while time.monotonic() < deadline:
                    status, body = _request(base_url + "/health", timeout=2)
                    health = json.loads(body) if status == 200 else {}
                    if (health.get("ready") and health.get("api_version") == 3
                            and not args.stub_codecs and not health.get("neural_available")):
                        raise RuntimeError(
                            "a classical-only demo server still owns the port; stop it with "
                            "scripts/wsl/stop_demo.ps1 and retry verification"
                        )
                    if (status == 200 and health.get("ready") and health.get("api_version") == 3
                            and (args.stub_codecs or health.get("neural_available") is True)):
                        break
                    time.sleep(2)
                else:
                    raise TimeoutError(
                        f"No ready DeltaShot server appeared on {base_url} within "
                        f"{args.startup_timeout}s; bind error: {bind_error}"
                    ) from bind_error
                print(f"Reusing ready demo server at {base_url}")
            if owns_server:
                server.daemon_threads = True
                threading.Thread(target=server.serve_forever, daemon=True).start()

                runner_errors = []
                def create_runner():
                    try:
                        demo.DemoHandler.runner = StubCodecRunner(demo) if args.stub_codecs else demo.CodecRunner()
                    except Exception:
                        runner_errors.append(traceback.format_exc())

                print(f"Loading demo codecs; startup timeout is {args.startup_timeout}s", flush=True)
                threading.Thread(target=create_runner, daemon=True).start()
                deadline = time.monotonic() + args.startup_timeout
                while time.monotonic() < deadline:
                    if runner_errors:
                        raise RuntimeError("codec startup failed:\n" + runner_errors[0])
                    status, body = _request(base_url + "/health", timeout=2)
                    current = json.loads(body) if status == 200 else {}
                    if current.get("ready") and (args.stub_codecs or current.get("neural_available") is True):
                        break
                    print("Waiting for demo server readiness...", flush=True)
                    time.sleep(2)
                else:
                    raise TimeoutError(f"demo server did not become ready within {args.startup_timeout}s")
        checks.append({"check": "server startup", "status": "PASS"})
        print("PASS server startup")

        manifest_path = ROOT / "website_screenshots" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"website pair manifest missing: {manifest_path}; run scripts/wsl/build_premade_demos.ps1 first")
        pair_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pairs = pair_manifest.get("pairs", [])
        if len(pairs) < 12:
            raise AssertionError(f"expected at least 12 real/synthetic demo pairs, found {len(pairs)}")
        real_versions = [pair for pair in pairs if pair.get("origin") == "real" and pair["id"].startswith(("python_", "bootstrap_"))]
        if not 5 <= len(real_versions) <= 8:
            raise AssertionError(f"expected 5-8 real version pairs, found {len(real_versions)}")
        if args.stub_codecs:
            _prepare_stub_premades(demo, pairs, fixture_dir)
        premade_path = (demo.PREMADE_ROOT / "manifest.json") if args.stub_codecs else ROOT / "results" / "premade_demos" / "manifest.json"
        if not premade_path.is_file():
            raise FileNotFoundError(f"premade codec manifest missing: {premade_path}")
        premade_ids = {row.get("id") for row in json.loads(premade_path.read_text(encoding="utf-8")).get("pairs", [])}
        missing_premade = [pair["id"] for pair in pairs if pair["id"] not in premade_ids]
        if missing_premade:
            raise AssertionError(f"premade codec results missing pairs: {missing_premade[:5]}")
        for pair in pairs:
            target_path = ROOT / "website_screenshots" / pair["target"]
            reference_path = ROOT / "website_screenshots" / pair["reference"]
            if not target_path.is_file() or not reference_path.is_file():
                raise FileNotFoundError(f"pair {pair['id']} has a missing image")
            verify_response(base_url, f"gallery pair: {pair['id']}", target_path,
                            reference_path, checks, True)

        tiny = {"target": {"filename": "tiny.png", "data": base64.b64encode(b"not an image").decode()},
                "reference": None}
        verify_invalid(base_url, "invalid image", tiny,
                       "target is not a readable PNG, JPEG, or WebP image", checks)
        empty = {"target": {"filename": "empty.png", "data": ""}, "reference": None}
        verify_invalid(base_url, "empty image", empty, "target image file is empty", checks)
        error_target = ROOT / "website_screenshots" / pairs[0]["target"]
        verify_error_contract(base_url, demo, error_target, checks, args.stub_codecs)

        if not args.skip_browser:
            target_path = ROOT / "website_screenshots" / pairs[0]["target"]
            browser_result = verify_browser(base_url, target_path, args.browser_channel)
            checks.append({"check": "Playwright upload flow", "status": "PASS", **browser_result})
            print(f"PASS Playwright upload flow: screenshots={PAGE_LIVE}, {PAGE_PREMADE}")
        else:
            checks.append({"check": "Playwright upload flow", "status": "SKIPPED",
                           "reason": "--skip-browser was specified"})
        write_reports(checks, stub=args.stub_codecs)
        print(f"PASS report: {REPORT_TEXT} and {REPORT_JSON}")
        return 0
    except Exception:
        error_text = traceback.format_exc()
        print("FAIL verification:\n" + error_text, file=sys.stderr)
        checks.append({"check": "verification run", "status": "FAIL", "error": error_text})
        write_reports(checks, error=error_text, stub=args.stub_codecs)
        return 1
    finally:
        if server is not None and owns_server:
            server.shutdown()
            server.server_close()
        if fixture_dir.exists():
            shutil.rmtree(fixture_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

"""Local upload-and-compress demo using actual codec bitstreams on CPU."""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.bucketing import bucket_from_ratio
from scripts.baselines.delta_png import compress_delta_png, decompress_delta_png
from scripts.baselines.jpeg_webp import compress_jpeg, compress_webp, decompress_jpeg, decompress_webp
from scripts.bitstream import (
    decode_compressai_bitstream,
    decode_mode_bitstream,
    read_mode_bitstream,
    read_neural_bitstream,
    write_mode_bitstream,
    write_neural_bitstream,
)
from scripts.demo_inputs import DemoInputError, WORKING_SIZE, load_upload, prepare_working_image
from scripts.mode_decision import select_actual_mode


ROOT = Path(__file__).resolve().parents[1]
WEBSITE_ROOT = ROOT / "website_screenshots"
PREMADE_ROOT = ROOT / "results" / "premade_demos"
PREMADE_RUNS = PREMADE_ROOT / "runs"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUT_ROOT = ROOT / "results" / "local_demo_runs"
CHART_ASSETS = {
    "bpp_by_bucket.png",
    "results_bpp_table.png",
    "mode_selection_check.png",
    "loss_curves.png",
}
MULTIPLE = 64
JPEG_QUALITY = 75
WEBP_QUALITY = 75
LAMBDA_VALUE = 0.0130
LOG = logging.getLogger("deltashot.local_demo")


def image_tensor(image: Image.Image, padded_size=None):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if padded_size is not None:
        padded_width, padded_height = padded_size
        pad_right = padded_width - image.width
        pad_bottom = padded_height - image.height
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    return tensor


def padded_dimensions(width, height):
    return (
        ((width + MULTIPLE - 1) // MULTIPLE) * MULTIPLE,
        ((height + MULTIPLE - 1) // MULTIPLE) * MULTIPLE,
    )


def tensor_image(tensor, original_size):
    width, height = original_size
    array = tensor[0, :, :height, :width].clamp(0, 1).mul(255).round()
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, "RGB")


def psnr_db(original: Image.Image, reconstruction: Image.Image):
    first = np.asarray(original.convert("RGB"), dtype=np.float64)
    second = np.asarray(reconstruction.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((first - second) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)


def change_stats(reference: Image.Image, target: Image.Image):
    first = np.asarray(reference.convert("RGB"), dtype=np.int16)
    second = np.asarray(target.convert("RGB"), dtype=np.int16)
    if first.shape != second.shape:
        raise ValueError("reference and target working images must have matching dimensions")
    ratio = float(np.any(np.abs(first - second) > 3, axis=2).mean())
    return {"changed_pixel_ratio": ratio, "bucket": bucket_from_ratio(ratio)}


def estimated_bpp(output, pixel_count):
    total_bits = sum(-torch.log(likelihood).sum() / math.log(2.0)
                     for likelihood in output["likelihoods"].values())
    return float(total_bits.item()) / pixel_count


def _compress_streams(output):
    streams = output["strings"]
    if len(streams) != 2 or any(len(stream) != 1 for stream in streams):
        raise ValueError("demo expects one RGB image per CompressAI call")
    return tuple(bytes(stream[0]) for stream in streams)


def _model_decompress(model, container, reference_tensor=None):
    decoded = decode_compressai_bitstream(model, container, reference_tensor)
    width, height = container["original_size"]
    image = tensor_image(decoded["x_hat"], (width, height))
    if image.size != (width, height):
        raise AssertionError(f"decoded dimensions {image.size} do not match {(width, height)}")
    return image


def encode_neural_file(model, image_tensor_value, path, original_size, codec, reference_tensor=None):
    with torch.inference_mode():
        if reference_tensor is None:
            output = model.compress(image_tensor_value)
        else:
            output = model.compress(reference_tensor, image_tensor_value)
        padded_height, padded_width = image_tensor_value.shape[-2:]
        original_w, original_h = original_size
        result = write_neural_bitstream(
            path, codec, (original_w, original_h), (padded_width, padded_height), output["shape"],
            _compress_streams(output),
        )
        likelihood_output = (model(image_tensor_value) if reference_tensor is None
                             else model(reference_tensor, image_tensor_value))
        return result, estimated_bpp(likelihood_output, original_w * original_h)


class CodecRunner:
    def __init__(self):
        from models.baseline_codec import get_baseline_codec
        from models.conditional_codec import ReferenceConditionedCodec

        torch.set_grad_enabled(False)
        LOG.info("loading No-Reference epoch-27 checkpoint: %s", CHECKPOINTS / "pipeline_noref_lambda_00130" / "checkpoint_epoch_27.pth")
        self.noref = self._load_model(
            get_baseline_codec(quality=3),
            CHECKPOINTS / "pipeline_noref_lambda_00130" / "checkpoint_epoch_27.pth",
        )
        LOG.info("No-Reference checkpoint loaded; preparing Reference-Conditioned codec")
        reference_model = ReferenceConditionedCodec(get_baseline_codec(quality=3))
        LOG.info("loading Reference-Conditioned epoch-27 checkpoint: %s", CHECKPOINTS / "pipeline_reference_lambda_00130" / "checkpoint_epoch_27.pth")
        self.reference = self._load_model(
            reference_model,
            CHECKPOINTS / "pipeline_reference_lambda_00130" / "checkpoint_epoch_27.pth",
        )
        LOG.info("both epoch-27 checkpoints loaded")

    @staticmethod
    def _load_model(model, checkpoint_path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"required epoch-27 checkpoint is missing: {checkpoint_path}")
        LOG.info("reading checkpoint file (%d bytes)", checkpoint_path.stat().st_size)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("epoch") != 27:
            raise ValueError(f"expected epoch 27 checkpoint, got {state.get('epoch')}: {checkpoint_path}")
        model.load_state_dict(state["model_state_dict"])
        LOG.info("checkpoint weights restored; updating entropy models")
        model.update(force=True)
        model.eval()
        return model.cpu()

    def run(self, target_image, target_upload_bytes, reference_image=None):
        run_id = uuid.uuid4().hex[:12]
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        uploaded_size = target_image.size
        target_image = prepare_working_image(target_image)
        working_size = target_image.size
        if working_size != WORKING_SIZE:
            raise AssertionError(f"working image must be {WORKING_SIZE}, got {working_size}")
        padded_size = padded_dimensions(*working_size)
        target = image_tensor(target_image, padded_size)
        reference = None
        if reference_image is not None:
            reference_image = prepare_working_image(reference_image)
            reference = image_tensor(reference_image, padded_size)
            pair_change = change_stats(reference_image, target_image)
        else:
            pair_change = None

        results = []
        results.append(self._result(
            run_id, "Original target (working image)", target_image, target_upload_bytes,
            note=(f"Upload {uploaded_size[0]} × {uploaded_size[1]} converted to "
                  f"{working_size[0]} × {working_size[1]}; codec sizes and PSNR use the working image."),
            psnr=math.inf, original_size=working_size,
        ))

        jpeg = compress_jpeg(target_image, JPEG_QUALITY)
        jpeg_path = run_dir / "target_q75.jpg"
        jpeg_path.write_bytes(jpeg.buffer)
        jpeg_decoded = decompress_jpeg(jpeg_path.read_bytes())
        results.append(self._result(
            run_id, "JPEG (quality 75)", jpeg_decoded, jpeg_path.stat().st_size,
            filename=jpeg_path.name, psnr=psnr_db(target_image, jpeg_decoded),
        ))

        webp = compress_webp(target_image, WEBP_QUALITY)
        webp_path = run_dir / "target_q75.webp"
        webp_path.write_bytes(webp.buffer)
        webp_decoded = decompress_webp(webp_path.read_bytes())
        results.append(self._result(
            run_id, "WebP (quality 75)", webp_decoded, webp_path.stat().st_size,
            filename=webp_path.name, psnr=psnr_db(target_image, webp_decoded),
        ))

        if reference is not None:
            delta = compress_delta_png(reference_image, target_image)
            delta_path = run_dir / "delta_png.bin"
            delta_path.write_bytes(delta.png_buffer)
            delta_decoded = decompress_delta_png(reference_image, delta_path.read_bytes())
            if delta_decoded.size != working_size:
                raise AssertionError(
                    f"Delta+PNG decoded dimensions {delta_decoded.size} do not match {working_size}"
                )
            results.append(self._result(
                run_id, "Delta+PNG (lossless)", delta_decoded, delta_path.stat().st_size,
                filename=delta_path.name, psnr=psnr_db(target_image, delta_decoded),
                note="Lossless path; requires the reference at decode time.", lossless=True,
                original_size=working_size,
            ))
        else:
            results.append(self._unavailable("Delta+PNG (lossless)", "Upload a reference to enable this path."))

        no_ref_path = run_dir / "noref.bin"
        _, noref_estimated = encode_neural_file(
            self.noref, target, no_ref_path, working_size, "noref",
        )
        no_ref_container = read_neural_bitstream(no_ref_path)
        noref_decoded = _model_decompress(self.noref, no_ref_container)
        noref_psnr = psnr_db(target_image, noref_decoded)
        noref_item = self._result(
            run_id, "No-Reference neural (lossy)", noref_decoded, no_ref_path.stat().st_size,
            filename=no_ref_path.name, psnr=noref_psnr,
            note="Epoch 27 checkpoint; real entropy-coded .bin, decoded from disk.",
            estimated_bpp=noref_estimated, original_size=working_size,
        )
        results.append(noref_item)

        ref_item = None
        if reference is not None:
            ref_path = run_dir / "reference_conditioned.bin"
            _, ref_estimated = encode_neural_file(
                self.reference, target, ref_path, working_size, "reference_conditioned", reference,
            )
            ref_container = read_neural_bitstream(ref_path)
            ref_decoded = _model_decompress(self.reference, ref_container, reference)
            ref_item = self._result(
                run_id, "Reference-Conditioned neural (lossy)", ref_decoded,
                ref_path.stat().st_size, filename=ref_path.name,
                psnr=psnr_db(target_image, ref_decoded),
                note="Epoch 27 checkpoint; requires the same reference at decode time.",
                estimated_bpp=ref_estimated, original_size=working_size,
            )
            results.append(ref_item)
        else:
            results.append(self._unavailable(
                "Reference-Conditioned neural (lossy)",
                "Upload a reference; the decoder needs the same side information.",
            ))

        if ref_item is not None:
            candidates = {
                "noref": (noref_item["bytes"], noref_decoded),
                "reference_conditioned": (ref_item["bytes"], ref_decoded),
            }
            selected_codec, mode_scores = select_actual_mode(
                candidates, target_image, LAMBDA_VALUE,
            )
            selected = noref_item if selected_codec == "noref" else ref_item
            selected_path = run_dir / "mode_selected.bin"
            selected_source = run_dir / selected["filename"]
            write_mode_bitstream(selected_path, selected_codec, selected_source.read_bytes())
            mode_container = read_mode_bitstream(selected_path)
            mode_decoded_tensor = decode_mode_bitstream(
                {"noref": self.noref, "reference_conditioned": self.reference},
                mode_container,
                reference,
            )
            width, height = mode_container["original_size"]
            mode_decoded = tensor_image(mode_decoded_tensor["x_hat"], (width, height))
            results.append(self._result(
                run_id, f"Mode Selection → {selected['label']}", mode_decoded,
                selected_path.stat().st_size, filename=selected_path.name,
                psnr=psnr_db(target_image, mode_decoded),
                note=(f"Chooses the lower actual-byte rate-distortion cost: actual bpp + "
                      f"{LAMBDA_VALUE:g} × MSE. The 5-byte mode header is included."),
                estimated_bpp=selected.get("estimated_bpp"), original_size=working_size,
            ))
            mode_manifest = {
                "status": "selected",
                "codec": selected_codec,
                "selected_file": selected_path.name,
                "objective": "actual bpp + lambda * RGB MSE on [0,1] working-image pixels",
                "lambda": LAMBDA_VALUE,
                "mode_header_bytes": 5,
                "candidate_objectives": mode_scores,
            }
        else:
            mode_manifest = {
                "status": "skipped_no_reference",
                "codec": None,
                "reason": "Mode Selection is skipped because no reference image was supplied.",
            }
            results.append(self._unavailable("Mode Selection", mode_manifest["reason"]))

        manifest = {
            "run_id": run_id,
            "original_size": list(uploaded_size),
            "working_size": list(working_size),
            "metric_size_basis": "all codec file-size rates and PSNR use the 512x512 working image",
            "reference_provided": reference is not None,
            "change_analysis": pair_change,
            "neural_checkpoints": {"no_reference_epoch": 27, "reference_conditioned_epoch": 27},
            "neural_decode_source": "each available neural decoder re-opened its written .bin file from disk",
            "neural_bitstreams": {
                "noref": {"file": no_ref_path.name, "decoded_from_disk": True},
                "reference_conditioned": (
                    {"file": ref_item["filename"], "decoded_from_disk": True}
                    if ref_item is not None else None
                ),
            },
            "neural_estimated_bpp_basis": "likelihood bits over 512x512 working-image pixels; excludes binary container bytes",
            "neural_actual_bpp_basis": "whole on-disk .bin bytes including container headers over 512x512 working-image pixels",
            "mode_selection": mode_manifest,
            "files": {path.name: path.stat().st_size for path in sorted(run_dir.iterdir()) if path.is_file()},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"run_id": run_id, "results": results, "manifest": manifest}

    @staticmethod
    def _result(run_id, label, image, size_bytes, filename=None, psnr=None,
                note="", lossless=False, estimated_bpp=None, original_size=None):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        decoded_filename = None
        if filename:
            decoded_filename = f"{Path(filename).stem}_decoded.png"
            (OUTPUT_ROOT / run_id / decoded_filename).write_bytes(buffer.getvalue())
        pixels = original_size[0] * original_size[1] if original_size else image.width * image.height
        source_upload = filename is None and (label.startswith("Input (") or label.startswith("Original target"))
        actual_bpp = (size_bytes * 8 / pixels) if pixels and not source_upload else None
        if estimated_bpp is not None and original_size:
            estimated_bpp_gap = actual_bpp - estimated_bpp
        else:
            estimated_bpp_gap = None
        return {
            "label": label,
            "bytes": int(size_bytes),
            "psnr_db": None if psnr is not None and math.isinf(psnr) else psnr,
            "psnr_label": "∞ dB (exact)" if psnr is not None and math.isinf(psnr) else f"{psnr:.2f} dB" if psnr is not None else "n/a",
            "actual_bpp": actual_bpp,
            "estimated_bpp": estimated_bpp,
            "actual_minus_estimated_bpp": estimated_bpp_gap,
            "lossless": lossless,
            "note": note,
            "image_data": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "file_url": f"/files/{run_id}/{filename}" if filename else None,
            "decoded_file_url": f"/files/{run_id}/{decoded_filename}" if decoded_filename else None,
            "filename": filename,
        }

    @staticmethod
    def _unavailable(label, note):
        return {"label": label, "available": False, "note": note}


class DemoHandler(BaseHTTPRequestHandler):
    runner = None
    classical_only = False
    inference_lock = threading.Lock()

    def log_message(self, format_string, *args):
        LOG.info("%s - %s", self.address_string(), format_string % args)

    def _send(self, status, content, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_error_json(self, status, reason):
        self._send(status, json.dumps({"error": str(reason)}).encode("utf-8"))

    def send_error(self, code, message=None, explain=None):
        reason = message or explain or self.responses.get(code, ("HTTP error",))[0]
        self._send_error_json(code, reason)

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if self.path == "/health":
            if self.runner is None and not self.classical_only:
                self._send(503, b'{"ready":false}')
            else:
                neural = "false" if self.classical_only else "true"
                epoch = "" if self.classical_only else ',"epoch":27'
                self._send(200, f'{{"ready":true,"device":"cpu","api_version":3,"neural_available":{neural}{epoch}}}'.encode())
            return
        if self.path == "/":
            page = (Path(__file__).with_name("local_demo.html")).read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if self.path == "/api/demo-data":
            self._serve_demo_data()
            return
        if self.path.startswith("/samples/"):
            self._serve_sample(self.path)
            return
        if self.path.startswith("/files/"):
            self._serve_file(self.path)
            return
        if self.path.startswith("/charts/"):
            self._serve_chart(self.path)
            return
        self._send_error_json(404, f"No route for {self.path}")

    def _serve_chart(self, request_path):
        filename = unquote(request_path.removeprefix("/charts/"))
        if filename not in CHART_ASSETS:
            self._send_error_json(404, "Requested chart is not available")
            return
        path = ROOT / "slide_assets" / filename
        if not path.is_file():
            self._send_error_json(404, f"Chart asset is missing: {filename}")
            return
        self._send(200, path.read_bytes(), "image/png")

    def _serve_file(self, request_path):
        parts = [unquote(part) for part in request_path.split("/")[2:]]
        if len(parts) != 2:
            self._send_error_json(404, "Requested file path is invalid")
            return
        roots = (OUTPUT_ROOT.resolve(), PREMADE_RUNS.resolve())
        path = next(((root / parts[0] / parts[1]).resolve() for root in roots
                     if (root / parts[0] / parts[1]).resolve().is_relative_to(root)
                     and (root / parts[0] / parts[1]).is_file()), None)
        if path is None:
            self._send_error_json(404, "Requested file does not exist")
            return
        mime = {
            ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")
        self._send(200, path.read_bytes(), mime)

    def _serve_sample(self, request_path):
        relative = unquote(request_path.removeprefix("/samples/"))
        root = WEBSITE_ROOT.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            self._send_error_json(404, "Requested demo screenshot does not exist")
            return
        self._send(200, path.read_bytes(), "image/png" if path.suffix.lower() == ".png" else "application/octet-stream")

    def _serve_demo_data(self):
        def read_json(path):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

        def epoch_curve(path):
            data = read_json(path) or {}
            by_epoch = {}
            for row in data.get("history", []):
                epoch, loss = row.get("epoch"), row.get("loss")
                if isinstance(epoch, int) and isinstance(loss, (int, float)) and math.isfinite(loss):
                    by_epoch[epoch] = loss
            return [{"epoch": epoch, "loss": loss} for epoch, loss in sorted(by_epoch.items())]

        eval_path = ROOT / "results" / "pipeline_eval_modes.jsonl"
        try:
            evaluation_rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()
                               if line.strip()]
        except (OSError, json.JSONDecodeError):
            evaluation_rows = []

        payload = {
            "pairs": read_json(WEBSITE_ROOT / "manifest.json"),
            "premade": read_json(PREMADE_ROOT / "manifest.json"),
            "summary": read_json(ROOT / "results" / "pipeline_summary.json"),
            "evaluation": evaluation_rows,
            "training": {
                "noref": epoch_curve(ROOT / "results" / "pipeline_noref_lambda_00130_metrics.json"),
                "reference": epoch_curve(ROOT / "results" / "pipeline_reference_lambda_00130_metrics.json"),
            },
            "premade_command": ".\\scripts\\wsl\\build_premade_demos.ps1",
        }
        self._send(200, json.dumps(payload, allow_nan=False).encode("utf-8"))

    def _classical_preview(self, payload):
        target, _ = load_upload(payload.get("target"), "target")
        original_size = list(target.size)
        target = prepare_working_image(target)
        reference = None
        if payload.get("reference"):
            reference, _ = load_upload(payload["reference"], "reference")
            reference = prepare_working_image(reference)
        run_id = uuid.uuid4().hex[:12]
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        def item(label, path, decoded, lossless=False):
            return CodecRunner._result(
                run_id, label, decoded, path.stat().st_size, filename=path.name,
                psnr=psnr_db(target, decoded), lossless=lossless, original_size=target.size,
            )

        jpeg = compress_jpeg(target, JPEG_QUALITY)
        webp = compress_webp(target, WEBP_QUALITY)
        jpeg_path, webp_path = run_dir / "target_q75.jpg", run_dir / "target_q75.webp"
        jpeg_path.write_bytes(jpeg.buffer)
        webp_path.write_bytes(webp.buffer)
        results = [
            item("JPEG (quality 75)", jpeg_path, decompress_jpeg(jpeg_path.read_bytes())),
            item("WebP (quality 75)", webp_path, decompress_webp(webp_path.read_bytes())),
        ]
        if reference is not None:
            delta_path = run_dir / "delta_png.bin"
            delta_path.write_bytes(compress_delta_png(reference, target).png_buffer)
            results.append(item("Delta+PNG (lossless)", delta_path,
                                decompress_delta_png(reference, delta_path.read_bytes()), True))
        else:
            results.append({"label": "Delta+PNG (lossless)", "available": False,
                            "note": "Upload a reference to enable this path."})
        return {"run_id": run_id, "original_size": original_size, "working_size": list(target.size),
                "change_analysis": change_stats(reference, target) if reference else None,
                "results": results, "neural_status": "Unavailable in this classical-only fallback. Build WSL premade results to enable learned codecs."}

    def do_POST(self):
        if self.path == "/api/shutdown":
            self._send(200, b'{"stopping":true}')
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path not in {"/api/compress", "/api/classical"}:
            self._send_error_json(404, f"No POST route for {self.path}")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise DemoInputError("request body is empty")
            if length > 48 * 1024 * 1024:
                self._send_error_json(413, "request body exceeds the 48 MiB limit")
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise DemoInputError("request body must contain valid JSON") from error
            if not isinstance(payload, dict):
                raise DemoInputError("request body must be a JSON object")
            if self.path == "/api/classical":
                self._send(200, json.dumps(self._classical_preview(payload), allow_nan=False).encode("utf-8"))
                return
            target, target_bytes = load_upload(payload.get("target"), "target")
            reference = None
            if payload.get("reference"):
                reference, _ = load_upload(payload["reference"], "reference")
            with self.inference_lock:
                if self.runner is None:
                    raise RuntimeError("neural codecs are unavailable in classical-only mode")
                result = self.runner.run(target, target_bytes, reference)
            self._send(200, json.dumps(result, allow_nan=False).encode("utf-8"))
        except DemoInputError as error:
            LOG.exception("local demo rejected an invalid request")
            self._send_error_json(400, str(error))
        except Exception as error:
            LOG.exception("local demo request failed")
            detail = str(error).strip() or f"{type(error).__name__} (no further detail)"
            self._send_error_json(500, f"Compression failed: {detail}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--classical-only", action="store_true",
                        help="serve the UI and classical fallback without importing CompressAI")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DemoHandler.classical_only = args.classical_only
    if args.classical_only:
        LOG.info("starting classical-only demo; neural codecs are unavailable")
        DemoHandler.runner = None
    else:
        LOG.info("loading epoch-27 neural checkpoints on CPU")
        DemoHandler.runner = CodecRunner()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    LOG.info("DeltaShot upload demo ready at http://127.0.0.1:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

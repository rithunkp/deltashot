import base64
import io

from PIL import Image, ImageDraw

from scripts import local_demo


def _upload(image, name):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"filename": name, "data": base64.b64encode(buffer.getvalue()).decode("ascii")}


def test_classical_only_preview_writes_real_files_for_1280x800_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(local_demo, "OUTPUT_ROOT", tmp_path)
    reference = Image.new("RGB", (1280, 800), "white")
    target = reference.copy()
    ImageDraw.Draw(target).rectangle((50, 50, 200, 140), fill="#176da5")

    response = local_demo.DemoHandler._classical_preview(
        None, {"target": _upload(target, "target.png"), "reference": _upload(reference, "reference.png")}
    )

    assert response["original_size"] == [1280, 800]
    assert response["working_size"] == [512, 512]
    assert response["change_analysis"]["changed_pixel_ratio"] > 0
    assert {item["label"] for item in response["results"]} == {
        "JPEG (quality 75)", "WebP (quality 75)", "Delta+PNG (lossless)",
    }
    for item in response["results"]:
        encoded = tmp_path / response["run_id"] / item["filename"]
        decoded = tmp_path / response["run_id"] / f"{encoded.stem}_decoded.png"
        assert encoded.stat().st_size == item["bytes"] > 0
        assert decoded.is_file()
        with Image.open(decoded) as image:
            assert image.size == (512, 512)

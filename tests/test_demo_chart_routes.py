from scripts import local_demo


def test_chart_route_serves_allowlisted_assets_only(tmp_path, monkeypatch):
    asset_dir = tmp_path / "slide_assets"
    asset_dir.mkdir()
    chart = asset_dir / "bpp_by_bucket.png"
    chart.write_bytes(b"chart png bytes")
    monkeypatch.setattr(local_demo, "ROOT", tmp_path)

    handler = local_demo.DemoHandler.__new__(local_demo.DemoHandler)
    sent = {}
    errors = {}
    handler._send = lambda status, body, content_type="application/json": sent.update(
        status=status, body=body, content_type=content_type
    )
    handler._send_error_json = lambda status, reason: errors.update(status=status, reason=reason)

    handler._serve_chart("/charts/bpp_by_bucket.png")
    assert sent == {"status": 200, "body": b"chart png bytes", "content_type": "image/png"}

    handler._serve_chart("/charts/../../PROJECT_STATE.md")
    assert errors["status"] == 404

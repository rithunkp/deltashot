import pytest

from scripts.matched_epoch import resolve_matched_checkpoints


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    return path


def test_epoch_checkpoint_paths_include_longer_final_noref(tmp_path):
    root = tmp_path
    noref = _touch(root / "checkpoints/pipeline_noref_lambda_00130/checkpoint_epoch_9.pth")
    ref = _touch(root / "checkpoints/pipeline_reference_lambda_00130/checkpoint_epoch_9.pth")
    final_noref = _touch(root / "checkpoints/pipeline_noref_lambda_00130/checkpoint_epoch_27.pth")

    resolved = resolve_matched_checkpoints(9, root=root)

    assert resolved == {"noref": noref, "reference": ref, "longer_noref": final_noref}


def test_explicit_checkpoint_paths_override_defaults(tmp_path):
    noref = _touch(tmp_path / "custom/no_reference.pth")
    ref = _touch(tmp_path / "custom/reference.pth")
    final_noref = _touch(tmp_path / "custom/final_noref.pth")

    resolved = resolve_matched_checkpoints(
        9, root=tmp_path, noref_checkpoint=noref, ref_checkpoint=ref,
        longer_noref_checkpoint=final_noref,
    )

    assert resolved["noref"] == noref
    assert resolved["reference"] == ref
    assert resolved["longer_noref"] == final_noref


def test_epoch_27_has_no_longer_trained_comparator(tmp_path):
    noref = _touch(tmp_path / "checkpoints/pipeline_noref_lambda_00130/checkpoint_epoch_27.pth")
    ref = _touch(tmp_path / "checkpoints/pipeline_reference_lambda_00130/checkpoint_epoch_27.pth")

    resolved = resolve_matched_checkpoints(27, root=tmp_path)

    assert resolved == {"noref": noref, "reference": ref, "longer_noref": None}


def test_negative_epoch_is_rejected():
    with pytest.raises(ValueError, match="zero-based"):
        resolve_matched_checkpoints(-1)

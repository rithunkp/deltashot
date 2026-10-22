"""Checkpoint selection helpers for matched-epoch evaluation."""
from pathlib import Path


FINAL_NOREF_EPOCH = 27


def resolve_matched_checkpoints(
    epoch,
    root=Path.cwd(),
    noref_checkpoint=None,
    ref_checkpoint=None,
    longer_noref_checkpoint=None,
):
    if epoch < 0:
        raise ValueError("epoch must be a zero-based non-negative index")

    root = Path(root)
    noref = Path(noref_checkpoint) if noref_checkpoint else (
        root / "checkpoints" / "pipeline_noref_lambda_00130" / f"checkpoint_epoch_{epoch}.pth"
    )
    reference = Path(ref_checkpoint) if ref_checkpoint else (
        root / "checkpoints" / "pipeline_reference_lambda_00130" / f"checkpoint_epoch_{epoch}.pth"
    )

    if longer_noref_checkpoint:
        longer = Path(longer_noref_checkpoint)
    elif epoch < FINAL_NOREF_EPOCH:
        longer = root / "checkpoints" / "pipeline_noref_lambda_00130" / f"checkpoint_epoch_{FINAL_NOREF_EPOCH}.pth"
    else:
        longer = None

    required = {"No-Reference": noref, "reference-conditioned": reference}
    if longer is not None:
        if epoch >= FINAL_NOREF_EPOCH:
            raise ValueError("the longer-trained No-Reference checkpoint must be later than the matched epoch")
        required["longer-trained No-Reference"] = longer
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("matched-epoch checkpoint(s) missing: " + "; ".join(missing))

    return {"noref": noref, "reference": reference, "longer_noref": longer}

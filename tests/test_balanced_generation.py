from __future__ import annotations

from scripts.generate_balanced_synthetic_pairs import next_pair_index, promote_candidate


def test_next_pair_index_skips_interrupted_run_ids(tmp_path) -> None:
    (tmp_path / "train" / "pair_000005").mkdir(parents=True)
    (tmp_path / "val" / "pair_000021").mkdir(parents=True)

    assert next_pair_index(tmp_path) == 22


def test_promote_candidate_copies_then_removes_only_staging_source(tmp_path) -> None:
    candidate = tmp_path / ".staging" / "train" / "candidate_00000000"
    candidate.mkdir(parents=True)
    for name in ("reference.png", "target.png", "meta.json"):
        (candidate / name).write_bytes(b"artifact")
    destination = tmp_path / "train" / "pair_000000"

    promote_candidate(candidate, destination)

    assert not candidate.exists()
    assert all((destination / name).is_file() for name in ("reference.png", "target.png", "meta.json"))

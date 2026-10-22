from __future__ import annotations

from random import Random

from scripts.generate_synthetic_pairs import NON_REPLACEMENT_OPERATORS, choose_chain, split_for_index


def test_split_assignment_is_deterministic() -> None:
    assert [split_for_index(index) for index in range(10)] == [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "val",
        "test",
    ]


def test_chain_schedule_covers_all_operators_and_lengths() -> None:
    chains = [choose_chain(index, Random(24041 + index)) for index in range(12)]

    assert {chain[0] for chain in chains if len(chain) == 1} == {*NON_REPLACEMENT_OPERATORS, "mutate_replace"}
    assert {len(chain) for chain in chains} == {1, 2, 4}
    assert chains[-1] == ("mutate_replace",)

#!/usr/bin/env python3
"""Unit tests for deterministic balanced training-position sampling."""

from __future__ import annotations

import unittest

from analyze_training_bias import (
    Candidate,
    deterministic_seeded_order,
    score_band,
    select_balanced_stratified_candidates,
    validate_stratified_coverage,
)


SCORES = {
    "-10..-5": -8,
    "0": 0,
    "+5..+10": 8,
}


def candidate(data_id: int, band: str, index: int, phase: int = 40) -> Candidate:
    return Candidate(
        priority=index,
        phase=phase,
        data_id=data_id,
        source_file=f"records{data_id}/0.dat",
        source_record=index,
        game_index=index,
        player=1 << ((data_id + index) % 64),
        opponent=0,
        player_color=0,
        policy=0,
        teacher=SCORES[band],
    )


def make_strata(
    data_ids: range,
    bands: tuple[str, ...] = tuple(SCORES),
    records_per_band: int = 2,
) -> dict[tuple[int, int, str], list[Candidate]]:
    result: dict[tuple[int, int, str], list[Candidate]] = {}
    for data_id in data_ids:
        for band_index, band in enumerate(bands):
            result[(40, data_id, band)] = [
                candidate(data_id, band, 1000 * data_id + 100 * band_index + index)
                for index in range(records_per_band)
            ]
    return result


class DeterministicBalancedSamplingTest(unittest.TestCase):
    def test_seeded_order_is_deterministic_and_seed_dependent(self) -> None:
        values = list(range(20))
        first = deterministic_seeded_order(values, 20260828, "ids")
        self.assertEqual(first, deterministic_seeded_order(values, 20260828, "ids"))
        self.assertNotEqual(first, deterministic_seeded_order(values, 20260829, "ids"))
        self.assertCountEqual(first, values)

    def test_outer_round_covers_every_id_before_a_second_pick(self) -> None:
        strata = make_strata(range(10, 16))
        selected = select_balanced_stratified_candidates(
            strata, phase=40, target=6, seed=20260828
        )
        self.assertEqual({item.data_id for item in selected}, set(range(10, 16)))
        coverage = validate_stratified_coverage(strata, 40, 6, selected)
        self.assertEqual(coverage["selected_data_id_count"], 6)

    def test_truncated_first_round_uses_distinct_seeded_ids(self) -> None:
        strata = make_strata(range(20, 30))
        selected = select_balanced_stratified_candidates(
            strata, phase=40, target=4, seed=20260828
        )
        self.assertEqual(len({item.data_id for item in selected}), 4)
        validate_stratified_coverage(strata, 40, 4, selected)

    def test_inner_round_covers_bands_before_repeating(self) -> None:
        strata = make_strata(range(7, 8), records_per_band=2)
        first_round = select_balanced_stratified_candidates(
            strata, phase=40, target=3, seed=20260828
        )
        self.assertEqual({score_band(item.teacher) for item in first_round}, set(SCORES))
        all_rows = select_balanced_stratified_candidates(
            strata, phase=40, target=6, seed=20260828
        )
        counts = {
            band: sum(score_band(item.teacher) == band for item in all_rows)
            for band in SCORES
        }
        self.assertEqual(set(counts.values()), {2})
        validate_stratified_coverage(strata, 40, 6, all_rows)

    def test_exhausted_id_does_not_block_other_ids(self) -> None:
        strata = make_strata(range(1, 3), bands=("0",), records_per_band=4)
        strata[(40, 1, "0")] = [candidate(1, "0", 1)]
        selected = select_balanced_stratified_candidates(
            strata, phase=40, target=5, seed=20260828
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(sum(item.data_id == 1 for item in selected), 1)
        self.assertEqual(sum(item.data_id == 2 for item in selected), 4)
        validate_stratified_coverage(strata, 40, 5, selected)

    def test_coverage_validator_rejects_old_low_id_bias(self) -> None:
        strata = make_strata(range(1, 4), bands=("0",), records_per_band=2)
        biased = [strata[(40, 1, "0")][0], strata[(40, 1, "0")][1]]
        with self.assertRaisesRegex(RuntimeError, "outer round-robin"):
            validate_stratified_coverage(strata, 40, 2, biased)

    def test_mismatched_stratum_key_is_rejected(self) -> None:
        wrong = {(40, 1, "0"): [candidate(1, "+5..+10", 1)]}
        with self.assertRaisesRegex(ValueError, "teacher band"):
            select_balanced_stratified_candidates(wrong, 40, 1, 20260828)


if __name__ == "__main__":
    unittest.main()

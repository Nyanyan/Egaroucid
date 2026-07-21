#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from strength_elo_confidence import (
    ClusteredEloData,
    analyze_clustered_elos,
    bca_interval,
    bootstrap_score_sums,
    fit_bootstrap_elos,
    fit_elos_from_pair_scores,
    load_clustered_elo_data,
    make_bootstrap_cluster_rows,
)
from strength_reporting import estimate_elos
from strength_tournament import (
    GameResult,
    MatchSetResult,
    PlayerSpec,
    TournamentStats,
)


def make_result(
    task_id: int,
    p0_idx: int,
    p1_idx: int,
    set_index: int,
    opening: str,
    score: float,
) -> MatchSetResult:
    disc_diff = 2 if score == 1.0 else -2 if score == 0.0 else 0
    games = tuple(
        GameResult(
            p0_idx=p0_idx,
            p1_idx=p1_idx,
            p0_is_black=p0_is_black,
            black_idx=p0_idx if p0_is_black else p1_idx,
            white_idx=p1_idx if p0_is_black else p0_idx,
            p0_disc_diff=disc_diff,
            black_stones=33 if disc_diff > 0 else 31,
            white_stones=31 if disc_diff > 0 else 33,
            transcript=opening,
        )
        for p0_is_black in (True, False)
    )
    return MatchSetResult(
        task_id=task_id,
        p0_idx=p0_idx,
        p1_idx=p1_idx,
        set_index=set_index,
        opening=opening,
        p0_disc_diff=float(disc_diff),
        p0_black_disc_diff=disc_diff,
        p0_white_disc_diff=disc_diff,
        color_games=games,
    )


class StrengthEloConfidenceTest(unittest.TestCase):
    names = ("p0", "p1", "p2")
    pairs = ((0, 1), (0, 2), (1, 2))
    scores = np.asarray(
        [
            [1.0, 1.0, 0.5],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 1.0],
        ]
    )

    def test_bootstrap_uses_one_shared_cluster_draw_for_every_pair(self) -> None:
        scores = np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
        sampled_rows = np.asarray([[0, 0, 2], [1, 1, 1]])
        sums = bootstrap_score_sums(scores, sampled_rows)
        np.testing.assert_array_equal(
            sums,
            np.asarray([[5.0, 50.0], [6.0, 60.0]]),
        )

    def test_bootstrap_seed_is_reproducible(self) -> None:
        first = make_bootstrap_cluster_rows(50, 100, 613)
        second = make_bootstrap_cluster_rows(50, 100, 613)
        different = make_bootstrap_cluster_rows(50, 100, 614)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))

    def test_point_estimator_matches_live_tournament_estimator(self) -> None:
        specs = [
            PlayerSpec(name, (name,), 2, False) for name in self.names
        ]
        stats = TournamentStats(len(specs))
        task_id = 0
        for set_index, scores in enumerate(self.scores):
            for (p0_idx, p1_idx), score in zip(self.pairs, scores):
                stats.record(
                    make_result(
                        task_id,
                        p0_idx,
                        p1_idx,
                        set_index,
                        f"opening_{set_index}",
                        float(score),
                    )
                )
                task_id += 1
        live = estimate_elos(specs, stats)
        fitted = fit_elos_from_pair_scores(
            self.names,
            self.pairs,
            np.sum(self.scores, axis=0),
            np.full(len(self.pairs), len(self.scores)),
        )
        np.testing.assert_allclose(
            fitted,
            [live[name] for name in self.names],
            rtol=0.0,
            atol=1.0e-9,
        )
        self.assertAlmostEqual(float(np.mean(fitted)), 1500.0)

    def test_bca_has_explicit_percentile_fallback(self) -> None:
        interval = bca_interval(
            1500.0,
            [1490.0, 1500.0, 1510.0, 1520.0],
            [1500.0, 1500.0, 1500.0],
        )
        self.assertEqual(
            interval.method,
            "cluster_percentile_bootstrap_fallback",
        )
        self.assertEqual(
            interval.fallback_reason,
            "jackknife acceleration is undefined",
        )
        self.assertLessEqual(interval.low, interval.high)

    def _write_synthetic_output(
        self,
        output_dir: Path,
        omit_last_result: bool = False,
    ) -> None:
        point_elos = fit_elos_from_pair_scores(
            self.names,
            self.pairs,
            np.sum(self.scores, axis=0),
            np.full(len(self.pairs), len(self.scores)),
        )
        total_match_sets = len(self.scores) * len(self.pairs)
        manifest = {
            "schema_version": 3,
            "experiment_id": "synthetic",
            "configuration": {
                "schedule": {
                    "match_sets_per_pair": len(self.scores),
                    "total_match_sets": total_match_sets,
                    "same_opening_sequence_for_every_pair": True,
                }
            },
        }
        aggregate = {
            "schema_version": 3,
            "experiment_id": "synthetic",
            "completed_match_sets": total_match_sets,
            "total_match_sets": total_match_sets,
            "completed_actual_games": 2 * total_match_sets,
            "total_actual_games": 2 * total_match_sets,
            "planned_match_sets_by_pair": [
                [
                    0 if player_idx == opponent_idx else len(self.scores)
                    for opponent_idx in range(len(self.names))
                ]
                for player_idx in range(len(self.names))
            ],
            "players": [{"name": name, "command": []} for name in self.names],
            "summary": [
                {
                    "name": name,
                    "paired_set_elo_descriptive": float(elo),
                }
                for name, elo in zip(self.names, point_elos)
            ],
        }
        (output_dir / "strength_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (output_dir / "strength_results.json").write_text(
            json.dumps(aggregate),
            encoding="utf-8",
        )
        rows = []
        task_id = 0
        for set_index, scores in enumerate(self.scores):
            for (p0_idx, p1_idx), score in zip(self.pairs, scores):
                result = make_result(
                    task_id,
                    p0_idx,
                    p1_idx,
                    set_index,
                    f"opening_{set_index}",
                    float(score),
                )
                rows.append(
                    {
                        "schema_version": 2,
                        "task": {
                            "task_id": task_id,
                            "p0_idx": p0_idx,
                            "p1_idx": p1_idx,
                            "set_index": set_index,
                            "opening": f"opening_{set_index}",
                        },
                        "result": result.to_dict(),
                    }
                )
                task_id += 1
        if omit_last_result:
            rows.pop()
        (output_dir / "strength_games.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_raw_loader_builds_complete_cluster_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            self._write_synthetic_output(output_dir)
            data = load_clustered_elo_data(output_dir)
        self.assertEqual(data.names, self.names)
        self.assertEqual(data.pairs, self.pairs)
        self.assertEqual(data.set_indices, (0, 1, 2, 3))
        np.testing.assert_array_equal(data.paired_scores, self.scores)

    def test_raw_loader_rejects_incomplete_xot_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            self._write_synthetic_output(output_dir, omit_last_result=True)
            with self.assertRaisesRegex(
                ValueError,
                "not a complete round robin",
            ):
                load_clustered_elo_data(output_dir)

    def test_analysis_is_seeded_and_centered(self) -> None:
        point_elos = fit_elos_from_pair_scores(
            self.names,
            self.pairs,
            np.sum(self.scores, axis=0),
            np.full(len(self.pairs), len(self.scores)),
        )
        data = ClusteredEloData(
            experiment_id="synthetic",
            names=self.names,
            pairs=self.pairs,
            set_indices=(0, 1, 2, 3),
            openings=("o0", "o1", "o2", "o3"),
            paired_scores=self.scores,
            raw_results_sha256="0" * 64,
            manifest_sha256="1" * 64,
            aggregate_sha256="2" * 64,
            existing_point_elos={
                name: float(elo) for name, elo in zip(self.names, point_elos)
            },
        )
        first = analyze_clustered_elos(
            data,
            bootstrap_replicates=20,
            bootstrap_seed=613,
            workers=1,
        )
        second = analyze_clustered_elos(
            data,
            bootstrap_replicates=20,
            bootstrap_seed=613,
            workers=1,
        )
        np.testing.assert_array_equal(
            first.bootstrap_cluster_rows,
            second.bootstrap_cluster_rows,
        )
        np.testing.assert_allclose(
            first.bootstrap_elos,
            second.bootstrap_elos,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.mean(first.bootstrap_elos, axis=1),
            1500.0,
            rtol=0.0,
            atol=1.0e-8,
        )

    def test_parallel_and_single_worker_fits_are_identical(self) -> None:
        sampled_rows = make_bootstrap_cluster_rows(4, 8, 613)
        score_sums = bootstrap_score_sums(self.scores, sampled_rows)
        single = fit_bootstrap_elos(
            self.names,
            self.pairs,
            score_sums,
            cluster_count=4,
            workers=1,
        )
        parallel = fit_bootstrap_elos(
            self.names,
            self.pairs,
            score_sums,
            cluster_count=4,
            workers=2,
        )
        np.testing.assert_array_equal(single, parallel)


if __name__ == "__main__":
    unittest.main()

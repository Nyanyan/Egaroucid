import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board_test_support import make_game
from game_dataset import (
    BOARD_DTYPE,
    infer_ordered_game_ids,
    load_game_dataset,
    load_manifest_positions,
    save_split_manifest,
    split_counts,
    split_game_positions,
)


class GameDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = [make_game(i, 2 + i % 7)[0] for i in range(40)]
        cls.samples = np.concatenate(cls.games)
        cls.ids = np.repeat(np.arange(40), [len(game) for game in cls.games])

    def assert_partitions(self, indices, expected_ids, expected_positions):
        position_sets = [set(indices[name]) for name in ("train", "val", "test")]
        game_sets = [set(expected_ids[indices[name]]) for name in ("train", "val", "test")]
        for sets in (position_sets, game_sets):
            for left, right in ((0, 1), (0, 2), (1, 2)):
                self.assertFalse(sets[left] & sets[right])
        self.assertEqual(set(expected_positions), set.union(*position_sets))

    def test_game_counts_membership_and_reproducibility(self):
        indices, summary = split_game_positions(self.samples, self.ids)
        self.assertEqual([36, 2, 2], [summary["splits"][name]["games"] for name in indices])
        self.assert_partitions(indices, self.ids, range(len(self.samples)))
        repeated, _ = split_game_positions(self.samples, self.ids)
        changed, _ = split_game_positions(self.samples, self.ids, seed=614)
        for name in indices:
            np.testing.assert_array_equal(indices[name], repeated[name])
        self.assertFalse(np.array_equal(indices["test"], changed["test"]))

    def test_ratio_rounding_and_validation(self):
        self.assertEqual((120763, 6709, 6709), split_counts(134181, .9, .05, .05))
        for ratios in ((.9, 0, .1), (.9, -.1, .2), (.9, float("nan"), .05), (float("inf"), .05, .05)):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                split_counts(100, *ratios)
        with self.assertRaisesRegex(ValueError, "too few games"):
            split_counts(19, .9, .05, .05)

    def test_ordered_games_across_file_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "0.dat", Path(tmp) / "1.dat"]
            self.samples[:3].tofile(paths[0])  # Split inside the second game.
            self.samples[3:].tofile(paths[1])
            samples, ids = load_game_dataset(paths)
        np.testing.assert_array_equal(samples, self.samples)
        np.testing.assert_array_equal(ids, self.ids)

    def test_full_games_with_passes_and_short_terminal_games(self):
        games = [make_game(seed)[0] for seed in (1, 2, 3)]
        ids = infer_ordered_game_ids(np.concatenate(games))
        np.testing.assert_array_equal(ids, np.repeat(np.arange(3), [len(game) for game in games]))

    def test_unrecoverable_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "initial position"):
            infer_ordered_game_ids(self.samples[1:])
        broken = self.samples.copy()
        broken[1] = self.games[-1][-1]
        with self.assertRaisesRegex(ValueError, "boundary"):
            infer_ordered_game_ids(broken)

    def test_explicit_ids_support_interleaving_and_directory_scoping(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for directory in ("records0", "records1"):
                root = Path(tmp) / directory
                root.mkdir()
                path = root / "0.dat"
                self.samples[1:5].tofile(path)
                np.array([10, 11, 10, 11], dtype="<u8").tofile(path.with_suffix(".game_ids"))
                paths.append(path)
            _, ids = load_game_dataset(paths)
        np.testing.assert_array_equal(ids, [0, 1, 0, 1, 2, 3, 2, 3])

    def test_missing_or_corrupt_sidecar_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "0.dat", Path(tmp) / "1.dat"]
            for path in paths:
                self.samples.tofile(path)
            self.ids.astype("<u8").tofile(paths[0].with_suffix(".game_ids"))
            with self.assertRaisesRegex(ValueError, "missing .game_ids"):
                load_game_dataset(paths)
            paths[1].with_suffix(".game_ids").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "count does not match"):
                load_game_dataset(paths)

    def test_position_cap_keeps_whole_games(self):
        cutoff = sum(len(game) for game in self.games[:25])
        indices, summary = split_game_positions(self.samples, self.ids, max_position_samples=cutoff + 1)
        self.assertEqual(25, summary["games"])
        self.assert_partitions(indices, self.ids, range(cutoff))
        with self.assertRaisesRegex(ValueError, "positive"):
            split_game_positions(self.samples, self.ids, max_position_samples=0)

    def test_invalid_labels_do_not_change_game_membership(self):
        samples = self.samples.copy()
        samples["policy"][0] = -1
        before, _ = split_game_positions(self.samples, self.ids)
        after, _ = split_game_positions(samples, self.ids)
        for name in before:
            np.testing.assert_array_equal(after[name], before[name][before[name] != 0])
        self.assert_partitions(after, self.ids, range(1, len(samples)))

    def test_sampling_only_subsets_each_assigned_partition(self):
        full, _ = split_game_positions(self.samples, self.ids)
        sampled, _ = split_game_positions(self.samples, self.ids, sample_caps={"train": 20, "val": 3})
        for name in full:
            self.assertTrue(set(sampled[name]) <= set(full[name]))
            self.assertEqual(len(sampled[name]), len(set(sampled[name])))
        self.assertEqual(20, len(sampled["train"]))
        self.assertEqual(3, len(sampled["val"]))
        np.testing.assert_array_equal(full["test"], sampled["test"])

    def test_manifest_reuses_membership_and_checks_data(self):
        indices, summary = split_game_positions(self.samples, self.ids)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "0.dat"
            self.samples.tofile(path)
            save_split_manifest(Path(tmp), [path], indices, summary)
            manifest = Path(tmp) / "dataset_split.json"
            for name in indices:
                selected, metadata = load_manifest_positions(manifest, [path], name)
                np.testing.assert_array_equal(indices[name], selected)
                self.assertEqual("game", metadata["unit"])
            with self.assertRaisesRegex(ValueError, "requires --data-split"):
                load_manifest_positions(manifest, [path], "all")
            changed = self.samples.copy()
            changed["score"][0] = 1
            changed.tofile(path)
            with self.assertRaisesRegex(ValueError, "data mismatch"):
                load_manifest_positions(manifest, [path], "test")

    def test_loaded_partitions(self):
        import policy_data

        indices, _ = split_game_positions(self.samples, self.ids)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "0.dat"
            self.samples.tofile(path)
            with contextlib.redirect_stdout(io.StringIO()):
                splits = policy_data.load_game_three_way_split([policy_data.BoardFile(path, len(self.samples))], .9, .05, .05, 613, None)
            for name, selected in indices.items():
                expected = sorted((int(self.samples[i]["player"]), int(self.samples[i]["opponent"]), int(self.samples[i]["policy"])) for i in selected)
                actual = sorted(zip(splits[name].player.tolist(), splits[name].opponent.tolist(), splits[name].policy.tolist()))
                self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


import policy_data
from board_test_support import make_game
from wthor_dataset import load_manifest_splits, infer_ordered_game_ids, save_split_manifest, split_game_positions


class PolicyDataTest(unittest.TestCase):
    def test_prepared_partitions_are_reused_without_splitting_again(self):
        samples = np.concatenate([make_game(i, 3 + i % 4)[0] for i in range(40)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "records1"
            source.mkdir()
            samples[:5].tofile(source / "0.dat")
            samples[5:].tofile(source / "1.dat")
            indices, summary = split_game_positions(samples, infer_ordered_game_ids(samples), seed=614)
            save_split_manifest(root / "prepared", sorted(source.glob("*.dat")), indices, summary)
            self.assertEqual([36, 2, 2], [summary["splits"][name]["games"] for name in ("train", "val", "test")])
            files = policy_data.discover_board_files(root, 1, 1)
            paths = [item.path for item in files]
            manifest = root / "prepared/dataset_split.json"
            indices, _ = load_manifest_splits(manifest, paths)
            ids = infer_ordered_game_ids(samples)
            games = [set(ids[indices[name]]) for name in ("train", "val", "test")]
            for left, right in ((0, 1), (0, 2), (1, 2)):
                self.assertFalse(games[left] & games[right])
            with mock.patch.object(policy_data, "split_game_positions", side_effect=AssertionError("must reuse prepared splits")), contextlib.redirect_stdout(io.StringIO()):
                splits = policy_data.load_game_three_way_split(files, .8, .1, .1, 999, None, output_dir=root / "trained", split_manifest=manifest)
            for name, selected in indices.items():
                expected = sorted((int(row["player"]), int(row["opponent"]), int(row["policy"])) for row in samples[selected])
                actual = sorted(zip(splits[name].player.tolist(), splits[name].opponent.tolist(), splits[name].policy.tolist()))
                self.assertEqual(expected, actual)
            saved = json.loads((root / "trained/dataset_split.json").read_text())
            self.assertEqual(614, saved["seed"])
            with self.assertRaisesRegex(ValueError, "caps"):
                policy_data.load_game_three_way_split(files, .9, .05, .05, 613, 100, split_manifest=manifest)
            changed = samples.copy()
            changed["score"][0] = 1
            changed[:5].tofile(source / "0.dat")
            with self.assertRaisesRegex(ValueError, "data mismatch"):
                load_manifest_splits(manifest, paths)


if __name__ == "__main__":
    unittest.main()

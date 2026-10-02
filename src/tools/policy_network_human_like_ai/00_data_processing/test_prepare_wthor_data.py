import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import policy_data
import prepare_wthor_data as prepare
from test_game_dataset import make_game
from wthor_dataset import load_manifest_splits, infer_ordered_game_ids


class PrepareWthorTest(unittest.TestCase):
    def test_default_source_is_wthor_and_explicit_path_needs_no_environment(self):
        with mock.patch.dict(os.environ, {"EGAROUCID_DATA": "data-root"}):
            self.assertEqual(Path("data-root/train_data/board_data/records1"), prepare.default_board_data_dir())
        with mock.patch.dict(os.environ, {}, clear=True):
            args = prepare.make_argparser().parse_args(["--board-data-dir", "records1"])
            self.assertEqual(Path("records1"), args.board_data_dir)
            with self.assertRaisesRegex(ValueError, "EGAROUCID_DATA"):
                prepare.default_board_data_dir()

    def test_prepared_partitions_are_reused_without_splitting_again(self):
        samples = np.concatenate([make_game(i, 3 + i % 4)[0] for i in range(40)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "records1"
            source.mkdir()
            samples[:5].tofile(source / "0.dat")
            samples[5:].tofile(source / "1.dat")
            args = prepare.make_argparser().parse_args([
                "--board-data-dir", str(source), "--output-dir", str(root / "prepared"),
                "--seed", "614",
            ])
            with mock.patch.dict(os.environ, {}, clear=True):
                summary = prepare.prepare(args)
            self.assertEqual("WTHOR", summary["source"])
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

    def test_training_defaults_and_preprocessing_location(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "10_train_policy_network"))
        import train_policy_network as train
        import run_training_experiments as runner
        with mock.patch.dict(os.environ, {}, clear=True):
            args = train.make_argparser().parse_args([])
            run_args = runner.make_argparser().parse_args([])
        for parsed in (args, run_args):
            self.assertEqual((1, 1), (parsed.board_data_index_start, parsed.board_data_index_end))
            self.assertIsNone(parsed.data_root)
        self.assertIs(train.load_game_three_way_split, policy_data.load_game_three_way_split)
        self.assertIs(train.position_samples_to_targets, policy_data.position_samples_to_targets)

    def test_experiment_runner_forwards_prepared_manifest(self):
        import run_training_experiments as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ["runner", "--configs", "8x1", "--split-manifest", str(root / "dataset_split.json"),
                    "--output-dir", str(root / "trained"), "--log-dir", str(root / "logs")]
            resource = {"returncode": 0, "elapsed_sec": 1, "max_main_memory_mib": 10, "resource_samples": 1}
            with mock.patch.object(sys, "argv", argv), mock.patch.object(runner, "run_monitored_command", return_value=resource) as run, contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            command = run.call_args.args[0]
            self.assertEqual(str(root / "dataset_split.json"), command[command.index("--split-manifest") + 1])
            self.assertEqual("1", command[command.index("--board-data-index-start") + 1])


if __name__ == "__main__":
    unittest.main()

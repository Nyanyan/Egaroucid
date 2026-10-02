import contextlib
import gc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import prepared_policy_data as data
import train_policy_network as train
import run_training_experiments as runner
from board_test_support import make_game
from pipeline_common import BoardWriter, write_manifest


class PreparedTrainingTest(unittest.TestCase):
    def setUp(self):
        train.enable_memory_growth()

    def fixture(self, root, overlap=False):
        counts = {}
        for index, name in enumerate(("train", "val", "test")):
            samples, _ = make_game(index + 10)
            writer = BoardWriter(root / name, 17)
            writer.write(samples, np.full(len(samples), 0 if overlap else index, dtype="<u8"))
            writer.close()
            counts[name] = {"positions": len(samples), "games": 1}
        write_manifest(root, "board_split_increased", "fixture", splits=counts, augmentation_factor=8)

    def test_environment_dataset_override_and_no_default_caps(self):
        with mock.patch.dict(os.environ, {"POLICY_DATA_ROOT": "data", "POLICY_DATASET": "other"}, clear=True):
            args = train.make_argparser().parse_args([])
            self.assertEqual(Path("data/board_split_increased/other").resolve(), data.resolve_board_split_dir(args))
            self.assertTrue(all(getattr(args, f"max_{s}_samples") is None for s in ("train", "val", "test")))
            args = train.make_argparser().parse_args(["--dataset", "override"])
            self.assertEqual("override", data.resolve_board_split_dir(args).name)
            args = train.make_argparser().parse_args(["--board-split-dir", "prepared"])
            self.assertEqual(Path("prepared"), data.resolve_board_split_dir(args))
            with self.assertRaisesRegex(ValueError, "single directory"):
                data.resolve_board_split_dir(train.make_argparser().parse_args(["--dataset", "../escape"]))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "POLICY_DATA_ROOT"):
                data.resolve_board_split_dir(train.make_argparser().parse_args([]))
        with mock.patch.dict(os.environ, {"EGAROUCID_DATA": "data"}, clear=True):
            self.assertEqual(Path("data/20261002_policy_network_human_like_data"), train.make_argparser().parse_args([]).data_root)

    def test_sharded_batches_caps_and_features_preserve_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            paths = sorted((root / "train").glob("*.dat"))
            expected = np.concatenate([np.fromfile(p, dtype=data.BOARD_DTYPE) for p in paths])
            split = data.PreparedSplit(paths)
            actual = np.concatenate(list(split.batches(13)))
            np.testing.assert_array_equal(actual, expected)
            shuffled = np.concatenate(list(split.batches(13, shuffle=True)))
            self.assertEqual(sorted(shuffled.tobytes()[i:i+19] for i in range(0, shuffled.nbytes, 19)),
                             sorted(expected.tobytes()[i:i+19] for i in range(0, expected.nbytes, 19)))
            sampled = data.PreparedSplit(paths, 21)
            same = data.PreparedSplit(paths, 21)
            np.testing.assert_array_equal(sampled.indices, same.indices)
            np.testing.assert_array_equal(sampled.read(0, 21), expected[sampled.indices])
            batches = list(sampled.as_dataset(8).as_numpy_iterator())
            self.assertEqual([8, 8, 5], [len(y) for x, y in batches])
            x, y = data.position_samples_to_targets(expected[sampled.indices])
            np.testing.assert_array_equal(np.concatenate([b[0] for b in batches]), x)
            np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), y)
            split.close()
            sampled.close()
            same.close()
            del split, sampled, same
            gc.collect()

    def test_shuffle_covers_multiple_blocks_and_last_partial_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = np.zeros(131079, dtype=data.BOARD_DTYPE)
            records["player"] = np.arange(len(records))
            paths = [Path(tmp) / f"{i}.dat" for i in range(3)]
            for path, chunk in zip(paths, np.array_split(records, 3)):
                chunk.tofile(path)
            split = data.PreparedSplit(paths)
            try:
                first = np.concatenate(list(split.batches(1000, shuffle=True, seed=1)))
                repeated = np.concatenate(list(split.batches(1000, shuffle=True, seed=1)))
                changed = np.concatenate(list(split.batches(1000, shuffle=True, seed=2)))
                np.testing.assert_array_equal(np.sort(first["player"]), records["player"])
                np.testing.assert_array_equal(first, repeated)
                self.assertFalse(np.array_equal(first, changed))
                self.assertEqual(132, len(list(split.batches(1000, shuffle=True))))
            finally:
                split.close()

    def test_experiment_runner_forwards_prepared_dataset_and_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ["runner", "--configs", "8x1", "--board-split-dir", "prepared",
                    "--dataset", "other", "--max-train-samples", "23", "--max-test-samples", "17",
                    "--output-dir", str(root / "run"), "--log-dir", str(root / "logs")]
            resource = {"returncode": 0, "elapsed_sec": 1, "max_main_memory_mib": 1, "resource_samples": 1}
            with mock.patch.object(sys, "argv", argv), mock.patch.object(runner, "run_monitored_command", return_value=resource) as run, contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            command = run.call_args.args[0]
            for option, value in (("--board-split-dir", "prepared"), ("--dataset", "other"),
                                  ("--max-train-samples", "23"), ("--max-test-samples", "17")):
                self.assertEqual(value, command[command.index(option) + 1])
            self.assertNotIn("--max-val-samples", command)
            self.assertNotIn("--split-mode", command)

    def test_manifest_checksum_and_game_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            self.fixture(root, overlap=True)
            with self.assertRaisesRegex(ValueError, "multiple prepared splits"):
                data.load_prepared_splits(root, {}, 613, Path(tmp) / "out")
            path = root / "test/0000000.dat"
            contents = bytearray(path.read_bytes())
            contents[-1] ^= 1
            path.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                data.load_prepared_splits(root, {}, 613, Path(tmp) / "out")
            gc.collect()

    def test_streamed_metrics_match_array_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            split = data.PreparedSplit(sorted((root / "test").glob("*.dat")))
            args = train.make_argparser().parse_args(["--configs", "8x1"])
            model = train.build_model(train.parse_model_specs(args.configs, args)[0])
            streamed = train.evaluate_policy_topn_arrays(model, "test", split, [1, 3], 23, 8)
            arrays = train.position_samples_to_split_arrays(split.read(0, split.positions))
            materialized = train.evaluate_policy_topn_arrays(model, "test", arrays, [1, 3], 23, 8)
            self.assertEqual(streamed["topn"], materialized["topn"])
            self.assertAlmostEqual(streamed["cross_entropy"], materialized["cross_entropy"], places=5)
            split.close()
            del split
            gc.collect()

    def test_train_select_then_test_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "dataset")
            events = []
            original_train = train.train_one_model
            original_eval = train.evaluate_policy_topn_arrays
            original_select = train.save_selected_model

            def tracked_train(*args):
                events.append("train")
                return original_train(*args)

            def tracked_eval(*args):
                events.append(args[1])
                return original_eval(*args)

            def tracked_select(*args):
                events.append("select")
                return original_select(*args)

            argv = ["train", "--board-split-dir", str(root / "dataset"), "--configs", "8x1,12x1",
                    "--epochs", "1", "--batch-size", "16", "--max-train-samples", "23",
                    "--max-val-samples", "17", "--output-dir", str(root / "run")]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(train, "train_one_model", side_effect=tracked_train), \
                    mock.patch.object(train, "evaluate_policy_topn_arrays", side_effect=tracked_eval), \
                    mock.patch.object(train, "save_selected_model", side_effect=tracked_select), contextlib.redirect_stdout(io.StringIO()):
                train.main()
            self.assertEqual(["train", "val", "train", "val", "select", "test"], events)
            result = json.loads((root / "run/test_metrics.json").read_text())
            manifest = json.loads((root / "dataset/manifest.json").read_text())
            self.assertEqual(manifest["splits"]["test"]["positions"], result["positions"])
            provenance = (root / "run/prepared_dataset.json").read_text()
            self.assertNotIn(str(root).replace("\\", "\\\\"), provenance)
            self.assertTrue((root / "run/selected_model.h5").exists())
            # Release call histories retaining memory maps before Windows cleanup.
            gc.collect()


if __name__ == "__main__":
    unittest.main()

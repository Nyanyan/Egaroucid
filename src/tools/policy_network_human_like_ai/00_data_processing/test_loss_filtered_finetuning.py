"""Verify mistake selection, all eight transforms, and held-out game separation."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import prepare_loss_filtered_finetuning as prepare
from board_symmetry import augment_positions
from pipeline_common import BOARD_DTYPE, BoardWriter, WEIGHT_FORMAT, source_info, write_manifest


class LossFilteredTests(unittest.TestCase):
    def fixture(self, root, *, overlap=False, bad_histogram=False):
        transcripts = root / "transcript_split/toy"
        transcripts.mkdir(parents=True)
        (transcripts / "manifest.json").write_text("{}", encoding="utf-8")
        splits = {}
        ordinary, weighted = root / "board_split/toy", root / "board_split/toy_LOSS_WEIGHTED"
        records = np.zeros(5, BOARD_DTYPE)
        records["player"], records["opponent"] = 0x814, 0x404000
        records["policy"] = [7, 8, 9, 10, 11]
        records["score"] = -128
        losses = np.array([0, 1, 2, 3, 128], dtype=np.uint8)
        for index, name in enumerate(("train", "val", "test")):
            ids = np.full(len(records), 11 if overlap else index + 11, dtype="<u8")
            for directory, weights in ((ordinary, False), (weighted, True)):
                writer = BoardWriter(directory / name, 3, weighted=weights)
                try:
                    writer.write(records, ids, np.exp2(losses.astype(float)/2) if weights else None)
                finally:
                    writer.close()
            splits[name] = dict(games=1, positions=len(records))
        metadata = dict(source=source_info(transcripts), split_unit="identical_transcript_group",
                        seed=613, ratios=dict(train=.9, val=.05, test=.05),
                        record_format="<QQbbb", game_ids_format="uint64", score_convention="terminal",
                        splits=splits)
        write_manifest(ordinary, "board_split", "toy", **metadata)
        write_manifest(weighted, "board_split", "toy_LOSS_WEIGHTED", **metadata,
                       analysis=dict(level=10, configuration="fixture"),
                       sample_weights=dict(format=WEIGHT_FORMAT, formula="2 ** (disc_loss / 2)",
                                           normalization="none in files", clipping=None))
        histogram = np.bincount(losses, minlength=129)
        if bad_histogram:
            histogram[0] += 1
        stats = dict(dataset="toy", source=source_info(transcripts), configuration="fixture",
                     loss_formula="max(played_score, alternative_score) - played_score",
                     distribution={name: dict(bins=[dict(count=int(n)) for n in histogram]) for name in splits})
        stats_path = root / "cache/toy_LOSS_WEIGHTED/statistics/loss_distribution.json"
        stats_path.parent.mkdir(parents=True)
        stats_path.write_text(json.dumps(stats), encoding="utf-8")
        return records, losses

    def arguments(self, root):
        return argparse.Namespace(data_root=root, dataset="toy", min_loss=2,
                                  output_dir=root/"prepared", positions_per_batch=2, progress_seconds=60)

    def test_selects_boundary_and_maintains_ids_through_all_transforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records, losses = self.fixture(root)
            manifest = prepare.run(self.arguments(root))
            expected, ids = augment_positions(records[losses >= 2], np.full(3, 11, dtype="<u8"))
            actual = np.fromfile(root/"prepared/train/0000000.dat", dtype=BOARD_DTYPE)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.fromfile(root/"prepared/train/0000000.game_ids", dtype="<u8"), ids)
            for name in ("val", "test"):
                np.testing.assert_array_equal(np.fromfile(root/f"prepared/{name}/0000000.dat", dtype=BOARD_DTYPE), records[losses >= 2])
            self.assertEqual(24, manifest["splits"]["train"]["positions"])
            self.assertEqual(3, manifest["splits"]["val"]["positions"])
            self.assertNotIn("sample_weights", manifest)
            self.assertFalse(list((root/"prepared").rglob("*.weights")))

    def test_rejects_overlapping_games_and_wrong_analysis_distribution(self):
        for option in ("overlap", "bad_histogram"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.fixture(root, **{option: True})
                with self.assertRaises(ValueError):
                    prepare.run(self.arguments(root))
                self.assertFalse((root/"prepared").exists())

    def test_all_integer_losses_decode_and_invalid_weight_encodings_fail(self):
        losses = np.arange(129)
        np.testing.assert_array_equal(prepare.decode_losses(np.exp2(losses/2)), losses)
        for weights in ([0], [float("nan")], [1.7], [2.**65]):
            with self.assertRaises(ValueError):
                prepare.decode_losses(weights)


if __name__ == "__main__":
    unittest.main()

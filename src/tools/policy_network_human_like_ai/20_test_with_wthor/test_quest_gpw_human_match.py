"""Check independent folder sampling, 30-player aggregation and safe cache reuse."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_quest_gpw_human_match as quest
import test_mistake_test_models as fixtures


def board_labels(records):
    return Counter(tuple(int(row[field]) for field in ("player", "opponent", "color", "policy"))
                   for row in records)


class QuestGPWTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = fixtures.MistakeTest()
        self.fixture.setUp()
        self.data = self.root / "data"
        self.source = {}
        for name, records in (("band_a", self.fixture.records[:18]), ("band_b", self.fixture.records[18:])):
            directory = self.data / name
            directory.mkdir(parents=True)
            # Include multiple .dat files and an unknown score as in unfinished Quest games.
            records = records.copy()
            records["score"] = -128
            records[:9].tofile(directory / "0000000.dat")
            records[9:].tofile(directory / "0000001.dat")
            self.source[name] = records

    def test_default_is_100000_per_folder(self):
        args = quest.make_argparser().parse_args([])
        self.assertEqual(100000, args.positions)
        self.assertEqual(613, args.sample_seed)
        self.assertIsNone(args.dataset)

    def test_sampling_is_reproducible_and_never_crosses_folder_boundaries(self):
        infos = quest.inspect_datasets(self.data, None, 11)
        self.assertEqual(["band_a", "band_b"], [info["name"] for info in infos])
        for info in infos:
            dataset, sample = quest.load_dataset(info, 613)
            repeated, repeated_sample = quest.load_dataset(info, 613)
            selected = np.sort(np.random.default_rng(613).choice(18, 11, replace=False))
            expected = self.source[info["name"]][selected]
            self.assertEqual(board_labels(expected), board_labels(dataset["records"]))
            self.assertEqual(11, len(dataset["records"]))
            self.assertEqual(sample, repeated_sample)
            np.testing.assert_array_equal(dataset["records"], repeated["records"])
            other_name = "band_b" if info["name"] == "band_a" else "band_a"
            self.assertFalse(set(board_labels(dataset["records"])) & set(board_labels(self.source[other_name])))

    def test_insufficient_data_unknown_folder_and_truncated_record_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "available 18"):
            quest.inspect_datasets(self.data, None, 100000)
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            quest.inspect_datasets(self.data, ["missing"], 11)
        selected = quest.inspect_datasets(self.data, ["band_b"], 11)
        self.assertEqual(["band_b"], [info["name"] for info in selected])
        with (self.data / "band_a/0000000.dat").open("ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            quest.inspect_datasets(self.data, None, 11)

    def test_small_worker_budget_covers_every_level_with_separate_caches(self):
        def collect(groups, workers, options):
            self.assertLessEqual(sum(min(len(o["levels"]), o["workers"]) for o in options.values()), workers)
            for engine, settings in options.items():
                self.assertEqual(quest.gpw.LEVELS, settings["levels"])
                self.assertEqual(self.root / "search" / engine, settings["output_dir"])
                self.assertEqual(1, settings["standalone_hint_count"])
            return {engine: (({level: {} for level in quest.gpw.LEVELS}, {}, {}, {}), 0)
                    for engine in options}

        files = {key: Path(key) for key in ("egaroucid_executable", "edax_executable", "edax_evaluation_data")}
        with patch.object(quest.gpw, "collect_comparison_hints", side_effect=collect), \
                patch.object(quest.gpw, "check_memory"):
            for workers in (2, 3, 8, 14, 32):
                hints, _ = quest.collect_dataset_hints(self.fixture.groups, self.root / "search", files, workers)
                self.assertEqual(14, len(hints))

    def test_two_folder_run_resume_and_changed_input_guard(self):
        args = quest.make_argparser().parse_args([
            "--board-data-root", str(self.data), "--positions", "11", "--workers", "2",
            "--output-dir", str(self.root / "output")])
        files = {name: self.root / name for name in
                 ("normal", "loss_a", "loss_b", "egaroucid_executable", "edax_executable", "edax_evaluation_data")}
        for path in files.values():
            path.write_bytes(b"fixture")
        resources = self.root / "resources"
        resources.mkdir()
        for name in ("eval.egev2", "eval_move_ordering_end.egev"):
            (resources / name).write_bytes(b"fixture")
        manifest = dict(experiment_id="test", configuration=dict(participants=dict(paper_roster=dict(random=dict(seed=57)))))
        hints = {(engine, level): self.fixture.hints[21]
                 for engine in ("egaroucid", "edax") for level in quest.gpw.LEVELS}
        status_path = args.output_dir / "run_status.json"
        with patch.object(quest.gpw, "load_tournament", return_value=(manifest, {}, files)), \
                patch.object(quest, "collect_dataset_hints", return_value=(hints, {})) as searches, \
                patch.object(quest.gpw, "predict_policy_logits", return_value=self.fixture.normal) as predictions:
            rows = quest.run(args)
            self.assertEqual(60, len(rows))
            self.assertTrue(all(row["positions"] == 11 for row in rows))
            self.assertEqual({"band_a", "band_b"}, {row["dataset"] for row in rows})
            self.assertEqual(2, searches.call_count)
            self.assertEqual(6, predictions.call_count)
            self.assertEqual("complete", quest.gpw.read_json(status_path)["state"])
            self.assertEqual(rows, quest.run(args))
            self.assertEqual(2, searches.call_count)
            self.assertEqual(6, predictions.call_count)
            for name in self.source:
                with np.load(args.output_dir / name / "predictions.npz") as saved:
                    self.assertEqual((11, 30), saved["predicted_policies"].shape)
                    self.assertTrue(set(board_labels(saved["records"])) <= set(board_labels(self.source[name])))
            with (self.data / "band_a/0000000.dat").open("r+b") as stream:
                stream.seek(18)  # score is ignored by agreement, but source identity still must change.
                stream.write(b"\x00")
            with self.assertRaisesRegex(ValueError, "conditions differ"):
                quest.run(args)


if __name__ == "__main__":
    unittest.main()

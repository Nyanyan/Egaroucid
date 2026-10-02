import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from board_replay import replay_games
from board_symmetry import TABLES, augment_positions, transform_bits
from pipeline_common import BOARD_DTYPE, SPLIT_NAMES, dataset_directory, iter_transcripts, load_manifest
from policy_data import calc_legal_batch
import policy_data
from policy_accuracy import TRANSFORM_MAPS, transform_bitboards
from raw_formats import read_wthor_csv
from test_game_dataset import make_game


def load_step(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEPS = [load_step(name) for name in ("01_convert_raw_data", "02_split_transcripts", "03_generate_board_data", "04_augment_board_data")]


class NumberedPipelineTest(unittest.TestCase):
    def test_csv_variants_keep_headerless_first_game(self):
        _, game = make_game(1)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            source.write_text(game + "\n" + game + "\n", encoding="utf-8-sig")
            self.assertEqual([(game, 1), (game, 2)], list(read_wthor_csv(source)))
            source.write_text('tournamentName,transcript\n"Cup, final",' + game.upper() + '\n', encoding="utf-8-sig")
            self.assertEqual([(game, 2)], list(read_wthor_csv(source)))

    def test_vector_replay_matches_reference_and_rejects_partial_games(self):
        games, expected = [], []
        for seed in range(30):
            rows, transcript = make_game(seed)
            games.append((seed, transcript))
            expected.append(rows)
        boards, ids, rejected = replay_games(games)
        self.assertEqual([], rejected)
        combined = np.concatenate(expected)
        for field in ("player", "opponent", "color", "policy"):
            np.testing.assert_array_equal(combined[field], boards[field])
        np.testing.assert_array_equal(ids, np.repeat(np.arange(30), [len(rows) for rows in expected]))
        boards, ids, rejected = replay_games([(31, "f5"), (32, "a1")])
        self.assertEqual(0, len(boards))
        self.assertEqual({31, 32}, {row["game_id"] for row in rejected})

    def test_symmetry_lookup_and_labels_preserve_legality(self):
        bits = np.array([1 << i for i in range(64)] + [0, (1 << 64) - 1], dtype=np.uint64)
        for mapping, tables in zip(TRANSFORM_MAPS, TABLES):
            np.testing.assert_array_equal(transform_bits(bits, tables), transform_bitboards(bits, mapping))
        samples, _ = make_game(11)
        expanded, ids = augment_positions(samples, np.full(len(samples), 11, dtype="<u8"))
        self.assertEqual(len(samples) * 8, len(expanded))
        np.testing.assert_array_equal(samples, expanded[::8])
        legal = calc_legal_batch(expanded["player"], expanded["opponent"])
        moves = np.left_shift(np.uint64(1), expanded["policy"].astype(np.uint64))
        self.assertTrue(np.all(legal & moves))
        self.assertTrue(np.all(ids == 11))
        np.testing.assert_array_equal(expanded["score"], np.repeat(samples["score"], 8))

    def test_full_pipeline_is_generic_and_split_membership_survives_augmentation(self):
        dataset = "another_dataset"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw_data" / dataset
            raw.mkdir(parents=True)
            games = [make_game(seed)[1] for seed in range(40)]
            (raw / "old.csv").write_text("\n".join(games[:20]) + "\n", encoding="utf-8")
            (raw / "new.csv").write_text("transcript,extra\n" + "\n".join(game + ',"a,b"' for game in games[20:] + [games[0]]) + "\n", encoding="utf-8")
            options = ["--data-root", str(root), "--dataset", dataset]
            results = []
            for i, step in enumerate(STEPS):
                extra = ["--format", "wthor-csv", "--games-per-file", "7"] if i == 0 else []
                if i == 2:
                    extra = ["--games-per-batch", "9", "--positions-per-file", "17"]
                if i == 3:
                    extra = ["--positions-per-batch", "13", "--positions-per-file", "101"]
                with contextlib.redirect_stdout(io.StringIO()):
                    results.append(step.run(step.make_argparser().parse_args(options + extra)))
            self.assertEqual(41, results[0]["games"])
            self.assertEqual(1, results[1]["duplicate_copies"])
            memberships = []
            transcripts_by_split = []
            for name in SPLIT_NAMES:
                assigned = list(iter_transcripts(root / "transcript_split" / dataset / name))
                game_ids = {g[0] for g in assigned}
                memberships.append(game_ids)
                transcripts_by_split.append({g[1] for g in assigned})
                for stage in ("board_split", "board_split_increased"):
                    directory = root / stage / dataset
                    manifest = load_manifest(directory, stage)
                    paths = sorted((directory / name).glob("*.dat"))
                    ids = np.concatenate([np.fromfile(p.with_suffix(".game_ids"), dtype="<u8") for p in paths])
                    self.assertEqual(game_ids, set(ids.tolist()))
                    self.assertEqual(manifest["splits"][name]["positions"], len(ids))
                self.assertEqual(results[2]["splits"][name]["positions"] * 8, results[3]["splits"][name]["positions"])
            for groups in (memberships, transcripts_by_split):
                for left, right in ((0, 1), (0, 2), (1, 2)):
                    self.assertFalse(groups[left] & groups[right])
            prepared = root / "board_split_increased" / dataset
            with mock.patch.object(policy_data, "split_game_positions", side_effect=AssertionError("must not re-split")):
                loaded = policy_data.load_prepared_board_splits(prepared, 613, root / "trained")
            for name in SPLIT_NAMES:
                self.assertEqual(results[3]["splits"][name]["positions"], len(loaded[name].policy))
            self.assertTrue((root / "trained/prepared_dataset.json").is_file())
            with self.assertRaises(FileExistsError):
                STEPS[0].run(STEPS[0].make_argparser().parse_args(options + ["--format", "wthor-csv"]))
            with contextlib.redirect_stdout(io.StringIO()):
                STEPS[1].run(STEPS[1].make_argparser().parse_args(options + ["--replace"]))
            for name, expected_ids in zip(SPLIT_NAMES, memberships):
                actual = {g[0] for g in iter_transcripts(root / "transcript_split" / dataset / name)}
                self.assertEqual(expected_ids, actual)
            self.assertEqual(1, len(list((root / "transcript_split").glob(dataset + ".backup-*"))))
            (root / "transcript" / dataset / "0000000.txt").write_text("f5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_manifest(root / "transcript" / dataset, "transcript")

    def test_dataset_path_cannot_escape_data_root(self):
        for name in ("../escape", "a/b", "a\\b", ".", "..", "E:escape"):
            with self.assertRaises(ValueError):
                dataset_directory(Path.cwd(), "raw_data", name)


if __name__ == "__main__":
    unittest.main()

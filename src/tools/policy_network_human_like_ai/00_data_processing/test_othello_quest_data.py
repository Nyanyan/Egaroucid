"""Check player ratings, forced passes, global splits and Quest output contracts."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from board_replay import replay_games
from board_test_support import make_game
from pipeline_common import BOARD_DTYPE, SPLIT_NAMES, iter_transcripts, load_manifest
from policy_data import load_prepared_board_splits
import prepare_othello_quest_data as quest


def write_game(directory, key, black_rating, white_rating, transcript):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"2026-10-04T08_20_22.540Z({key})"
                        f"black-123({black_rating})-white_456({white_rating}).oth")
    path.write_text(transcript.upper() + "\n", encoding="utf-8-sig")
    return path


def run_quietly(root, *extra):
    args = quest.make_argparser().parse_args(["--data-root", str(root), *extra])
    with contextlib.redirect_stdout(io.StringIO()):
        return quest.run(args)


def read_boards(directory):
    paths = sorted(directory.glob("*.dat"))
    if not paths:
        return np.empty(0, dtype=BOARD_DTYPE), np.empty(0, dtype="<u8")
    return (np.concatenate([np.fromfile(p, dtype=BOARD_DTYPE) for p in paths]),
            np.concatenate([np.fromfile(p.with_suffix(".game_ids"), dtype="<u8") for p in paths]))


class QuestDataTest(unittest.TestCase):
    def test_legacy_semicolon_and_pass_format_matches_compact_moves(self):
        self.assertEqual(quest.normalize_quest_transcript(";F5;D6;-;C3;\n"), "f5d6c3")
        for value in (";f5;invalid", ";f5;;d6", "f5-d6", ";-;"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    quest.normalize_quest_transcript(value)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_game(root / "a", "same", 900, 1100, "f5d6")
            legacy = write_game(root / "b", "same", 900, 1100, "f5d6")
            legacy.write_text(";F5;D6", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                games, _ = quest.read_games(root)
            self.assertEqual(len(games), 1)
            self.assertEqual(len(games[0].source_files), 2)

    def test_filename_ratings_and_half_open_bands(self):
        meta = quest.parse_filename("2026-10-04T08_20_22.540Z(123456)black-123(799)-white_456(1200).oth")
        self.assertEqual((meta["quest_game_id"], meta["black_rating"], meta["white_rating"]),
                         ("123456", 799, 1200))
        expected = {0: "under_800", 799: "under_800", 800: "800-1000", 999: "800-1000",
                    1000: "1000-1200", 1199: "1000-1200", 1200: "1200-1400", 2800: "2800-3000"}
        for rating, band in expected.items():
            self.assertEqual(quest.rating_band(rating), band)
        self.assertEqual(quest.band_bounds("under_800"), (None, 800))
        with self.assertRaisesRegex(ValueError, "black/white ratings"):
            quest.parse_filename("unknown(1000).oth")

    def test_duplicate_ids_require_matching_metadata_and_transcripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_game(root / "a", "same", 900, 1100, "f5")
            duplicate = write_game(root / "b", "same", 900, 1100, "f5")
            with contextlib.redirect_stdout(io.StringIO()):
                games, inputs = quest.read_games(root)
            self.assertEqual((len(games), len(inputs), len(games[0].source_files)), (1, 2, 2))
            duplicate.write_text("f5d6", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicting copies"):
                quest.read_games(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_game(root / "a", "same", 900, 1100, "f5")
            write_game(root / "b", "same", 901, 1100, "f5")
            with self.assertRaisesRegex(ValueError, "conflicting copies"):
                quest.read_games(root)

    def test_unfinished_opt_in_keeps_moves_but_never_illegal_prefixes(self):
        scalar, transcript = make_game(613)
        games = [(7, transcript), (8, transcript[:40]), (9, "f5f5")]
        boards, ids, rejected = replay_games(games, allow_unfinished=True)
        partial = boards[ids == 8]
        for name in ("player", "opponent", "color", "policy"):
            np.testing.assert_array_equal(partial[name], scalar[:20][name])
        self.assertTrue(np.all(partial["score"] == -128))
        self.assertEqual({row["game_id"] for row in rejected}, {9})
        default_boards, default_ids, default_rejected = replay_games(games)
        np.testing.assert_array_equal(boards[ids == 7], default_boards)
        self.assertEqual(set(default_ids), {7})
        self.assertEqual({row["game_id"] for row in default_rejected}, {8, 9})

    def test_global_split_and_color_routing_match_independent_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw_data"
            expected = {}
            ratings = [(799, 800), (999, 1000), (1199, 1200), (2800, 600), (1000, 1199)]
            pass_count = 0
            for seed in range(40):
                rows, transcript = make_game(seed)
                br, wr = ratings[seed % len(ratings)]
                key = f"game{seed:03d}"
                write_game(raw / "arbitrary_original_folder", key, br, wr, transcript)
                expected[key] = (rows, br, wr)
                pass_count += int(np.count_nonzero(rows["color"][1:] == rows["color"][:-1]))
            self.assertGreater(pass_count, 0)
            rows, transcript = make_game(0)
            write_game(raw / "second_folder", "game000", 799, 800, transcript)
            write_game(raw / "second_folder", "other_id_same_moves", 1200, 2800, transcript)
            expected["other_id_same_moves"] = (rows, 1200, 2800)
            write_game(raw / "second_folder", "partial", 799, 1200, transcript[:40])
            expected["partial"] = (rows[:20], 799, 1200)
            write_game(raw / "second_folder", "illegal", 800, 999, "a1")
            before = {path: path.read_bytes() for path in raw.rglob("*.oth")}
            result = run_quietly(root, "--games-per-batch", "7", "--positions-per-file", "17")
            self.assertEqual((result["raw_files"], result["games"], result["duplicate_file_copies"]), (44, 43, 1))
            self.assertEqual(sum(c["rejected_games"] for c in result["splits"].values()), 1)
            output = root / "othello_quest_split"
            meta = [json.loads(line) for line in (output / "transcript_split/all/games.jsonl").read_text().splitlines()]
            by_key = {entry["quest_game_id"]: entry for entry in meta}
            by_id = {entry["game_id"]: entry for entry in meta}
            self.assertEqual(by_key["game000"]["split"], by_key["other_id_same_moves"]["split"])
            memberships = {}
            saved = 0
            for band, splits in result["bands"].items():
                directory = output / "board_split" / band
                manifest = load_manifest(directory, "board_split")
                self.assertEqual(manifest["splits"], splits)
                for split in SPLIT_NAMES:
                    boards, ids = read_boards(directory / split)
                    self.assertEqual(len(boards), splits[split]["positions"])
                    self.assertEqual(len(np.unique(ids)), splits[split]["games"])
                    saved += len(boards)
                    for game_id in np.unique(ids):
                        entry = by_id[int(game_id)]
                        memberships.setdefault(int(game_id), set()).add(split)
                        self.assertEqual(entry["split"], split)
                        rows, br, wr = expected[entry["quest_game_id"]]
                        selected = np.array([quest.rating_band(br if c == 0 else wr) == band for c in rows["color"]])
                        for field in ("player", "opponent", "color", "policy"):
                            np.testing.assert_array_equal(boards[ids == game_id][field], rows[selected][field])
                        if entry["quest_game_id"] == "partial":
                            self.assertTrue(np.all(boards[ids == game_id]["score"] == -128))
            self.assertTrue(all(len(parts) == 1 for parts in memberships.values()))
            self.assertEqual(saved, sum(len(rows) for rows, _, _ in expected.values()))
            self.assertEqual(saved, sum(c["positions"] for c in result["splits"].values()))
            self.assertGreater(sum(c["positions"] for c in result["bands"]["under_800"].values()), 0)
            self.assertEqual(load_manifest(output, "othello_quest_split")["games"], 43)
            self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))
            rejected = [json.loads(line) for line in (output / "rejected_games.jsonl").read_text().splitlines()]
            self.assertEqual([entry["quest_game_id"] for entry in rejected], ["illegal"])
            with self.assertRaises(FileExistsError):
                run_quietly(root)
            replacement = run_quietly(root, "--replace", "--complete-games-only")
            self.assertEqual(sum(c["rejected_games"] for c in replacement["splits"].values()), 2)
            self.assertEqual(sum(c["positions"] for c in replacement["splits"].values()), saved - 20)
            self.assertEqual(len(list(root.glob("othello_quest_split.backup-*"))), 1)

    def test_compatible_with_training_loader_augmentation_and_repeatable_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in range(40):
                write_game(root / "raw_data" / "anything", f"g{seed:03d}", 800, 1200, make_game(seed)[1])
            summary = run_quietly(root)
            output = root / "othello_quest_split"
            for band in ("800-1000", "1200-1400"):
                loaded = load_prepared_board_splits(output / "board_split" / band, 613)
                for name in SPLIT_NAMES:
                    self.assertEqual(len(loaded[name].policy), summary["bands"][band][name]["positions"])
            spec = importlib.util.spec_from_file_location("augment", Path(__file__).with_name("04_augment_board_data.py"))
            augment = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(augment)
            args = augment.make_argparser().parse_args(["--data-root", str(output), "--dataset", "800-1000"])
            with contextlib.redirect_stdout(io.StringIO()):
                expanded = augment.run(args)
            for name in SPLIT_NAMES:
                self.assertEqual(expanded["splits"][name]["positions"], summary["bands"]["800-1000"][name]["positions"] * 8)
            rerun = root / "another_output"
            run_quietly(root, "--output-dir", str(rerun), "--games-per-batch", "3")
            for name in SPLIT_NAMES:
                self.assertEqual(list(iter_transcripts(output / "transcript_split/all" / name)),
                                 list(iter_transcripts(rerun / "transcript_split/all" / name)))

    def test_input_output_overlap_is_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw_data"
            raw.mkdir()
            for destination in (root, raw, raw / "output"):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(ValueError, "must not contain"):
                        run_quietly(root, "--output-dir", str(destination))

    def test_same_range_extraction_preserves_global_ids_splits_and_partial_moves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = [(600, 799), (800, 999), (1000, 1199), (1200, 1399),
                     (1400, 1599), (1600, 1799), (1800, 1999), (2000, 2199), (2200, 2700)]
            transcripts = [make_game(seed)[1] for seed in range(40)]
            for bi, (black, white) in enumerate(pairs):
                for seed, transcript in enumerate(transcripts):
                    write_game(root / "raw_data", f"same_{bi}_{seed:03d}", black, white, transcript)
                write_game(root / "raw_data", f"partial_{bi}", black, white, transcripts[0][:40])
            for bi, (black, white) in enumerate([(799, 800), (999, 1000), (2199, 2200)]):
                write_game(root / "raw_data", f"cross_{bi}", black, white, transcripts[0])
            run_quietly(root)
            source = root / "othello_quest_split/transcript_split/all"
            original = {row["game_id"]: row for row in map(json.loads,
                (source / "games.jsonl").read_text(encoding="utf-8").splitlines())}
            result = run_quietly(root, "--same-range-source", str(source), "--games-per-file", "7")
            self.assertEqual(len(result["datasets"]), 9)
            spec = importlib.util.spec_from_file_location("same_range_board_step",
                Path(__file__).with_name("03_generate_board_data.py"))
            board_step = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(board_step)
            all_output_ids = set()
            memberships = {}
            for dataset, counts in result["datasets"].items():
                self.assertEqual(counts["games"], 41)
                band = dataset.removeprefix("OQ_samerange_")
                transcript_dir = root / "transcript" / dataset
                unsplit = load_manifest(transcript_dir, "transcript")
                self.assertEqual(unsplit["games"], 41)
                extracted = list(iter_transcripts(transcript_dir))
                self.assertEqual(len(extracted), 41)
                split_dir = root / "transcript_split" / dataset
                split_manifest = load_manifest(split_dir, "transcript_split")
                self.assertEqual(split_manifest["inherited_split_source"], result["source"])
                for name in SPLIT_NAMES:
                    rows = list(iter_transcripts(split_dir / name))
                    self.assertEqual(len(rows), counts["splits"][name]["games"])
                    for gid, transcript, _, _ in rows:
                        entry = original[gid]
                        self.assertEqual(entry["split"], name)
                        self.assertEqual(entry["transcript"], transcript)
                        self.assertEqual(quest.same_range_band(entry["black_rating"]), band)
                        self.assertEqual(quest.same_range_band(entry["white_rating"]), band)
                        self.assertEqual(memberships.setdefault(transcript, name), name)
                        self.assertNotIn(gid, all_output_ids)
                        all_output_ids.add(gid)
                options = ["--data-root", str(root), "--dataset", dataset,
                           "--allow-unfinished", "--positions-per-file", "101"]
                with contextlib.redirect_stdout(io.StringIO()):
                    generated = board_step.run(board_step.make_argparser().parse_args(options))
                load_manifest(root / "board_split" / dataset, "board_split")
                for name in SPLIT_NAMES:
                    actual, ids = read_boards(root / "board_split" / dataset / name)
                    self.assertEqual(len(actual), counts["splits"][name]["positions"])
                    self.assertEqual(generated["splits"][name]["rejected_games"], 0)
                    assigned = [(gid, transcript) for gid, transcript, _, _ in iter_transcripts(split_dir / name)]
                    expected, expected_ids, rejected = replay_games(assigned, allow_unfinished=True)
                    self.assertEqual(rejected, [])
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(ids, expected_ids)
                self.assertEqual(sum(c["unfinished_positions"] for c in generated["splits"].values()), 20)
            self.assertEqual(len(all_output_ids), 9 * 41)
            self.assertTrue(all(not original[gid]["quest_game_id"].startswith("cross_") for gid in all_output_ids))
            # The ordinary board command must still reject unfinished games by default.
            args = board_step.make_argparser().parse_args(["--data-root", str(root),
                "--dataset", "OQ_samerange_under_800", "--replace"])
            with contextlib.redirect_stdout(io.StringIO()):
                default = board_step.run(args)
            self.assertEqual(sum(c["rejected_games"] for c in default["splits"].values()), 1)
            self.assertEqual(sum(c["positions"] for c in default["splits"].values()),
                             result["datasets"]["OQ_samerange_under_800"]["positions"] - 20)
            with self.assertRaises(FileExistsError):
                run_quietly(root, "--same-range-source", str(source))


if __name__ == "__main__":
    unittest.main()

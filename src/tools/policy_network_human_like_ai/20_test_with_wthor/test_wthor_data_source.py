from contextlib import redirect_stderr
import importlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wthor_data_source import default_board_data_dir, resolve_board_data_args
from wthor_human_match_evaluation import BoardState, load_position_groups
from evaluate_wthor_human_match import BOARD_DTYPE


class DataSourceTest(unittest.TestCase):
    def test_policy_root_takes_precedence_and_dataset_is_selectable(self):
        with patch.dict(os.environ, {"POLICY_DATA_ROOT": "prepared", "EGAROUCID_DATA": "old"}, clear=True):
            self.assertEqual(
                Path("prepared/board_split/another_dataset/test"),
                default_board_data_dir("another_dataset"),
            )

    def test_egaroucid_root_fallback_uses_prepared_test_split(self):
        with patch.dict(os.environ, {"EGAROUCID_DATA": "data"}, clear=True):
            self.assertEqual(
                Path("data/20261002_policy_network_human_like_data/board_split/WTHOR_1977_2024/test"),
                default_board_data_dir(),
            )

    def test_all_entry_points_resolve_selected_dataset_or_explicit_override(self):
        modules = [
            "wthor_human_match_experiment", "evaluate_wthor_human_match",
        ]
        for name in modules:
            with self.subTest(entry_point=name):
                module = importlib.import_module(name)
                positional = ["2"] if name == "wthor_human_match_experiment" else []
                if name == "evaluate_wthor_human_match":
                    positional = ["--weights", "unused.bin"]
                with patch.dict(os.environ, {"POLICY_DATA_ROOT": "prepared"}, clear=True):
                    parser = module.make_argparser()
                    args = parser.parse_args(positional + ["--dataset", "selected"])
                    resolve_board_data_args(args, parser)
                    self.assertEqual(Path("prepared/board_split/selected/test"), args.board_data_dir)
                with patch.dict(os.environ, {}, clear=True):
                    parser = module.make_argparser()
                    args = parser.parse_args(positional + ["--board-data-dir", "chosen"])
                    resolve_board_data_args(args, parser)
                    self.assertEqual(Path("chosen"), args.board_data_dir)

    def test_missing_environment_and_invalid_dataset_report_cli_errors(self):
        from wthor_human_match_experiment import make_argparser
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()):
            parser = make_argparser()
            with self.assertRaises(SystemExit):
                resolve_board_data_args(parser.parse_args(["2"]), parser)
            with self.assertRaises(SystemExit):
                parser.parse_args(["2", "--dataset", "../train"])

    def test_prepared_test_data_is_not_split_again(self):
        from wthor_human_match_experiment import make_argparser
        state = BoardState.initial()
        player, opponent = state.player_opponent_bits(state.side)
        records = np.zeros(20, dtype=BOARD_DTYPE)
        records["player"], records["opponent"] = player, opponent
        records["color"] = state.side
        records["policy"] = state.legal_policies(state.side)[0]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"POLICY_DATA_ROOT": directory}, clear=True,
        ):
            parser = make_argparser()
            args = parser.parse_args(["7"])
            resolve_board_data_args(args, parser)
            args.board_data_dir.mkdir(parents=True)
            path = args.board_data_dir / "0000000.dat"
            records.tofile(path)
            groups, available, _, _ = load_position_groups(
                [path], 20, args.positions, args.data_split, args.split_seed, args.sample_seed,
            )
        self.assertEqual(20, available)
        self.assertEqual(7, sum(group.sample_count for group in groups))

    def test_default_random_sample_is_100000_unique_reproducible_test_rows(self):
        from wthor_human_match_experiment import make_argparser
        from evaluate_wthor_blend_human_match import choose_data_split_positions
        args = make_argparser().parse_args([])
        self.assertEqual(args.positions, 100000)
        self.assertEqual(args.data_split, "all")
        self.assertEqual(args.sample_seed, 613)
        def sample(seed):
            return choose_data_split_positions(401545, args.data_split, .8, .1, .1,
                                               args.split_seed, args.positions, seed)
        selected, available = sample(args.sample_seed)
        self.assertEqual(available, 401545)
        self.assertEqual(len(selected), 100000)
        self.assertEqual(len(np.unique(selected)), 100000)
        self.assertGreaterEqual(selected.min(), 0)
        self.assertLess(selected.max(), available)
        self.assertGreater(selected.max(), 100000)  # Not a prefix of the input.
        np.testing.assert_array_equal(selected, sample(args.sample_seed)[0])
        self.assertFalse(np.array_equal(selected, sample(args.sample_seed + 1)[0]))

    def test_explicit_position_count_still_samples(self):
        from wthor_human_match_experiment import make_argparser
        self.assertEqual(7, make_argparser().parse_args(["7"]).positions)


if __name__ == "__main__":
    unittest.main()

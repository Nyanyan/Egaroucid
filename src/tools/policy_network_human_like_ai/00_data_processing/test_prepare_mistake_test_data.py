"""Full transcript/cache/board alignment, cache misses and input mutation checks."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

import prepare_mistake_test_data as step
from board_replay import replay_games
from miss_analysis import AnalysisCache
from test_miss_board_data import analysis_text, make_source


def board_step():
    spec = importlib.util.spec_from_file_location("mistake_fixture_boards", Path(__file__).with_name("03_generate_board_data.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareMistakeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.games = make_source(self.root)
        boards = board_step()
        with contextlib.redirect_stdout(io.StringIO()):
            boards.run(boards.make_argparser().parse_args([
                "--data-root", str(self.root), "--dataset", "WTHOR_1977_2024"]))
        self.cache_dir = self.root / "cache" / "WTHOR_1977_2024_LOSS_WEIGHTED"
        cache = AnalysisCache(self.cache_dir, {"level": 10})
        for game_id, transcript in self.games["test"]:
            replayed, _, _ = replay_games([(game_id, transcript)])
            losses = [i % 8 for i in range(len(replayed))]
            cache.put(transcript, analysis_text(transcript, replayed["color"].tobytes(), losses), 0)
        cache.close()
        self.output = self.root / "output"
        self.args = step.make_argparser().parse_args([
            "--data-root", str(self.root), "--output-dir", str(self.output)])

    def test_test_only_uses_cache_without_requiring_train_or_val_analyses(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = step.run(self.args)
        self.assertEqual([1, 2, 4, 6], result["thresholds"])
        for row in result["datasets"]:
            directory = self.output / row["dataset"]
            self.assertFalse((directory / "train").exists())
            records, ids = step.read_test_records(directory / "test")
            self.assertEqual(row["positions"], len(records))
            self.assertEqual({self.games["test"][0][0]}, set(ids))
            self.assertEqual(sum(i % 8 >= row["minimum_loss"] for i in range(result["source_positions"])), len(records))

    def test_missing_analysis_never_publishes_partial_dataset(self):
        cache = AnalysisCache(self.cache_dir, {"level": 10})
        cache.db.execute("DELETE FROM analyses")
        cache.db.commit()
        cache.close()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "missing analyze"):
            step.run(self.args)
        self.assertFalse(self.output.exists())

    def test_modified_test_record_fails_checksum_validation(self):
        path = self.root / "board_split" / "WTHOR_1977_2024" / "test" / "0000000.dat"
        contents = bytearray(path.read_bytes())
        contents[0] ^= 1
        path.write_bytes(contents)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "checksum mismatch"):
            step.run(self.args)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

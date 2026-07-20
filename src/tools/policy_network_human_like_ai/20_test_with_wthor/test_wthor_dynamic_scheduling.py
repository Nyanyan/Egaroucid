#!/usr/bin/env python3

import queue
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_wthor_blend_human_match import (
    BOARD_DTYPE,
    PositionSampleReader,
    iter_position_task_queue,
)


class DynamicSchedulingTest(unittest.TestCase):
    def test_position_task_queue_stops_at_sentinel(self) -> None:
        task_queue = queue.Queue()
        for item in (7, 2, 11, None, 99):
            task_queue.put(item)

        self.assertEqual([7, 2, 11], list(iter_position_task_queue(task_queue)))

    def test_position_reader_maps_global_positions_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = np.zeros(2, dtype=BOARD_DTYPE)
            second = np.zeros(3, dtype=BOARD_DTYPE)
            first["player"] = [10, 11]
            second["player"] = [20, 21, 22]
            first_path = root / "0.dat"
            second_path = root / "1.dat"
            first.tofile(first_path)
            second.tofile(second_path)

            reader = PositionSampleReader([first_path, second_path], 4)
            try:
                self.assertEqual(10, int(reader.get(0)["player"]))
                self.assertEqual(11, int(reader.get(1)["player"]))
                self.assertEqual(20, int(reader.get(2)["player"]))
                self.assertEqual(21, int(reader.get(3)["player"]))
                with self.assertRaises(IndexError):
                    reader.get(4)
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()

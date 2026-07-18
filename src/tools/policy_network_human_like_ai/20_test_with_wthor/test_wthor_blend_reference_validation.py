#!/usr/bin/env python3

import argparse
import sqlite3
import tempfile
import unittest
from pathlib import Path

from run_random_wthor_blend_experiment import make_console_reference_validation


def raw_hint(moves: list[str]) -> str:
    lines = ["| Level | Depth | Move | Score | Time | Nodes | NPS |"]
    for rank, move in enumerate(moves):
        lines.append(f"| 21 | 10 | {move} | {3 - rank} | 0 | 1 | 1 |")
    return "\n".join(lines)


def create_cache(path: Path, raw: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE hint_scores (
                key TEXT PRIMARY KEY,
                scores_json TEXT NOT NULL,
                raw_hint TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO hint_scores VALUES (?, ?, ?, ?)",
            (
                "level=21:hint=3:0000000000000001:0000000000000002:0",
                "{}",
                raw,
                0.0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


class ConsoleReferenceValidationTest(unittest.TestCase):
    def test_top3_set_equivalence_is_separate_from_exact_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            blend_cache = temp / "blend.sqlite3"
            console_cache = temp / "console.sqlite3"
            create_cache(blend_cache, raw_hint(["f5", "d3", "c4"]))
            create_cache(console_cache, raw_hint(["f5", "c4", "d3"]))

            result = make_console_reference_validation(
                argparse.Namespace(egaroucid_level=21),
                {"hint_cache_db": str(blend_cache)},
                [
                    {
                        "alpha": 0.0,
                        "positions": 1,
                        "top1_hits": 1,
                        "top3_hits": 1,
                    }
                ],
                [
                    {
                        "summary": {
                            "level": 21,
                            "positions": 1,
                            "top1_hits": 1,
                            "top3_hits": 1,
                        },
                        "result": {"hint_cache_db": str(console_cache)},
                    }
                ],
            )

        self.assertTrue(result["comparison_available"])
        self.assertTrue(result["evaluation_equivalent"])
        self.assertFalse(result["exact_hint_order_equal"])
        self.assertEqual(
            1,
            result["cached_output_comparison"]["top1_move_equal_states"],
        )
        self.assertEqual(
            1,
            result["cached_output_comparison"]["top3_set_equal_states"],
        )
        self.assertEqual(
            0,
            result["cached_output_comparison"]["top3_order_equal_states"],
        )


if __name__ == "__main__":
    unittest.main()

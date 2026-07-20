#!/usr/bin/env python3

import unittest

from evaluate_model_size_experiment import (
    TOP_N,
    assert_single_agreement_schema,
    dat_file_sort_key,
    parse_configs,
    selection_key,
    topn_map,
)


class ModelSizeExperimentTest(unittest.TestCase):
    def test_parse_configs(self) -> None:
        self.assertEqual(
            [(256, 4), (512, 6)],
            parse_configs("256x4,512X6"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_configs("256x4,256x4")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_configs("0x4")

    def test_dat_files_use_evaluator_numeric_order(self) -> None:
        from pathlib import Path

        ordered = sorted(
            [Path("10.dat"), Path("other.dat"), Path("2.dat")],
            key=dat_file_sort_key,
        )
        self.assertEqual(
            [Path("2.dat"), Path("10.dat"), Path("other.dat")],
            ordered,
        )

    def test_topn_schema_rejects_duplicates(self) -> None:
        rows = [
            {
                "top_n": n,
                "hits": n,
                "positions": 10,
                "accuracy": n / 10,
            }
            for n in TOP_N
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            topn_map({"positions": 10, "topn": rows + [rows[0]]})

    def test_removed_agreement_fields_are_rejected(self) -> None:
        result = {
            "agreement_definition": {"metric": "board_symmetry_aware"},
            "topn": [{"top_n": 1, "exact_accuracy": 0.5}],
        }
        with self.assertRaisesRegex(ValueError, "removed agreement fields"):
            assert_single_agreement_schema(result)

    def test_selection_uses_validation_agreement(self) -> None:
        larger = {
            "params": 200,
            "validation": {
                "topn": {
                    "1": {"accuracy": 0.5},
                    "3": {"accuracy": 0.8},
                }
            },
        }
        smaller = {
            "params": 100,
            "validation": {
                "topn": {
                    "1": {"accuracy": 0.5},
                    "3": {"accuracy": 0.8},
                }
            },
        }
        self.assertGreater(selection_key(smaller), selection_key(larger))


if __name__ == "__main__":
    unittest.main()

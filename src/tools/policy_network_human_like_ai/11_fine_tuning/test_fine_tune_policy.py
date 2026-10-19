"""Check augmentation boundaries, input identity and exclusive run ownership."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
import fine_tune_policy as fine
from policy_data import BOARD_DTYPE


class FineTuningTests(unittest.TestCase):
    def test_augmented_batches_cover_every_transform_with_exact_cardinality(self):
        # An asymmetric board makes accidental copies or label-only transforms visible.
        records = np.zeros(211, dtype=BOARD_DTYPE)
        records["player"] = 0x814
        records["opponent"] = 0x404000
        records["policy"] = 7
        records["score"] = -128
        class Source:
            positions = len(records)
            def read(self, start, end):
                return records[start:end]
        expected, _ = fine.augment_positions(records, np.arange(len(records)))
        for size in (64, 129, 512):
            for shuffle in (False, True):
                split = fine.AugmentedTrain(Source())
                batches = list(split.batches(size, shuffle=shuffle, seed=57))
                self.assertEqual((len(expected) + size - 1) // size, len(batches))
                self.assertTrue(all(len(batch) == size for batch in batches[:-1]))
                actual = np.concatenate(batches)
                self.assertEqual(Counter(row.tobytes() for row in expected),
                                 Counter(row.tobytes() for row in actual))
                again = np.concatenate(list(split.batches(size, shuffle=shuffle, seed=57)))
                np.testing.assert_array_equal(actual, again)

    def test_two_process_owners_cannot_share_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with fine.run_lock(directory):
                with self.assertRaises(OSError):
                    with fine.run_lock(directory):
                        pass
            with fine.run_lock(directory):
                pass

    def test_nonpositive_training_settings_fail_before_reading_inputs(self):
        for option, value in (("--epochs", "0"), ("--learning-rate", "nan"),
                              ("--batch-size", "0"), ("--max-train-samples", "-1")):
            args = fine.make_argparser().parse_args(["--output-dir", "unused", option, value])
            with self.assertRaises(ValueError):
                fine.inspect_inputs(args)


if __name__ == "__main__":
    unittest.main()

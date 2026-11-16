"""Check sample alignment, paper labels and both reversed-axis variants."""
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_disc_loss_agreement as plot
from prepare_disc_loss_plot_data import merge_wthor_predictions


class DiscLossAgreementTests(unittest.TestCase):
    def test_local_alpha_labels_match_paper_families(self):
        self.assertEqual(plot.point_label("alpha_1.2", .6), "α=0.2")
        self.assertEqual(plot.point_label("beta_l1_0.2", .6), "α=0.2")
        self.assertEqual(plot.point_label("alpha_2.0", .6), "α=1.0（石損モデル）")
        self.assertIn("66.60%", plot.point_label("alpha_0.8", .66597))
        self.assertEqual(plot.point_label("egaroucid_l17", .6), "Lv. 17")

    def test_mismatched_wthor_samples_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(2):
                path = Path(directory) / f"sample{i}.npz"
                np.savez(path, records=np.array([1+i, 2]), game_ids=np.array([1, 2]),
                         symmetry_target_masks=np.array([1, 1], dtype=np.uint64),
                         model_names=np.array(["alpha_0.0"]), predicted_policies=np.zeros((2, 1), dtype=np.int8))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "different rows"):
                merge_wthor_predictions(paths, ["alpha_0.0"], None)

    def test_zero_loss_coordinate_is_constant_in_mean_draws(self):
        draws = plot.mean_draws(np.array([0., 2.]), np.diag([0., .01]), np.random.default_rng(1), 100)
        self.assertTrue(np.all(draws[:, 0] == 0))
        self.assertGreater(draws[:, 1].std(), 0)
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            plot.mean_draws(np.array([0., 2.]), np.diag([-1., .01]), np.random.default_rng(1), 100)

    def test_both_variants_reverse_both_axes_and_filter_zebra_only(self):
        names = list(dict.fromkeys(name for spec in plot.CURVES.values() for name in spec["names"]))
        rows = [dict(model=name, mean_disc_loss=.1*i, wthor_accuracy=.6) for i, name in enumerate(names)]
        data = dict(rows=rows, references=[], covariance_model_order=names,
                    covariance_x=(np.eye(len(names))*.001).tolist(),
                    covariance_y=(np.eye(len(names))*.00001).tolist())
        tubes = plot.confidence_tubes(data, replicates=1000)
        self.assertIn("alpha_0.0", tubes["egaroucid"]["model_ids"])
        for with_zebra, expected in ((True, 31), (False, 27)):
            fig, axes, annotations = plot.make_figure(data, tubes, with_zebra)
            self.assertEqual(len(annotations), expected)
            self.assertTrue(all(axis.xaxis_inverted() for axis in axes.values()))
            self.assertEqual(any("zebra_" in key for key in annotations), with_zebra)
            self.assertIn("main/random", annotations)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()

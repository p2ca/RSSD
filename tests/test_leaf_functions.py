"""Behavioural checks for the leaf layer of :mod:`rssd`.

These run anywhere: they need neither reservoir data nor trained checkpoints. Their job is
to catch a migration that imports cleanly but silently lost a dependency or a code path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rssd import metrics, common                                    # noqa: E402
from rssd.data import reservoirs                                    # noqa: E402
from rssd.objectives import alignment, schedules                    # noqa: E402
from rssd.io import summary                                         # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_corr_is_one_for_a_perfect_linear_relation(self):
        x = np.arange(50, dtype=np.float64)
        pearson, spearman = metrics.corr(x, 2.0 * x + 1.0)
        self.assertAlmostEqual(pearson, 1.0, places=10)
        self.assertAlmostEqual(spearman, 1.0, places=10)

    def test_corr_ignores_non_finite_pairs(self):
        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, np.inf])
        pearson, _ = metrics.corr(x, y)
        self.assertAlmostEqual(pearson, 1.0, places=10)

    def test_rankdata_averages_ties(self):
        ranks = metrics._rankdata_average_ties(np.array([10.0, 20.0, 20.0, 30.0]))
        np.testing.assert_allclose(ranks, [1.0, 2.5, 2.5, 4.0])

    def test_per_reservoir_r2_is_one_for_exact_predictions(self):
        rng = np.random.default_rng(0)
        targets = rng.normal(size=(40, 2, 7))
        idx = {0: "AAA", 1: "BBB"}
        scores, worst = metrics._per_reservoir_r2(targets.copy(), targets, idx)
        for name in ("AAA", "BBB"):
            self.assertAlmostEqual(float(scores[name]), 1.0, places=8)
        self.assertEqual([name for name, _ in worst], ["AAA", "BBB"])

    def test_per_reservoir_r2_ranks_the_weakest_reservoir_first(self):
        rng = np.random.default_rng(2)
        targets = rng.normal(size=(60, 2, 7))
        preds = targets.copy()
        preds[:, 1, :] += rng.normal(scale=2.0, size=(60, 7))     # spoil the second reservoir
        scores, worst = metrics._per_reservoir_r2(preds, targets, {0: "AAA", 1: "BBB"})
        self.assertLess(scores["BBB"], scores["AAA"])
        self.assertEqual(worst[0][0], "BBB")

    def test_per_reservoir_r2_daily_returns_one_value_per_lead_day(self):
        rng = np.random.default_rng(1)
        targets = rng.normal(size=(30, 1, 7))
        daily = metrics._per_reservoir_r2_daily(targets.copy(), targets, {0: "AAA"})
        self.assertEqual(len(daily["AAA"]), 7)
        np.testing.assert_allclose(np.asarray(daily["AAA"], dtype=float), np.ones(7), atol=1e-8)


class CommonTests(unittest.TestCase):
    def test_set_seed_makes_torch_and_numpy_reproducible(self):
        common.set_seed(123)
        a = (torch.randn(4), np.random.rand(4))
        common.set_seed(123)
        b = (torch.randn(4), np.random.rand(4))
        self.assertTrue(torch.equal(a[0], b[0]))
        np.testing.assert_array_equal(a[1], b[1])

    def test_squeeze_pred_tensor_drops_a_trailing_singleton_axis(self):
        squeezed = common._squeeze_pred_tensor(torch.zeros(5, 7, 1))
        self.assertEqual(tuple(squeezed.shape), (5, 7))
        kept = common._squeeze_pred_tensor(torch.zeros(5, 7))
        self.assertEqual(tuple(kept.shape), (5, 7))


class ReservoirOrderTests(unittest.TestCase):
    def test_idx_to_reservoir_inverts_the_encode_map(self):
        idx = reservoirs._build_idx_to_reservoir({"AAA": 0, "BBB": 1})
        self.assertEqual(idx[0], "AAA")
        self.assertEqual(idx[1], "BBB")

    def test_reservoir_order_follows_node_index(self):
        order = reservoirs._reservoir_order_from_encode_map({"BBB": 1, "AAA": 0}, 2)
        self.assertEqual(list(order), ["AAA", "BBB"])


class AlignmentTests(unittest.TestCase):
    def test_mmd_between_identical_batches_is_near_zero(self):
        torch.manual_seed(0)
        z = torch.randn(64, 16)
        self.assertLess(float(alignment._mmd_rbf(z, z.clone())), 1e-5)

    def test_mmd_grows_when_the_two_batches_are_shifted_apart(self):
        torch.manual_seed(0)
        z = torch.randn(64, 16)
        near = float(alignment._mmd_rbf(z, z + 0.05))
        far = float(alignment._mmd_rbf(z, z + 5.0))
        self.assertGreater(far, near)

    def test_coral_between_identical_batches_is_near_zero(self):
        torch.manual_seed(0)
        z = torch.randn(128, 8)
        self.assertLess(float(alignment._coral_loss(z, z.clone())), 1e-6)

    def test_gradient_reversal_flips_the_sign_of_the_gradient(self):
        x = torch.ones(3, requires_grad=True)
        alignment._grad_reverse(x, 1.0).sum().backward()
        torch.testing.assert_close(x.grad, -torch.ones(3))

    def test_dann_loss_is_finite_and_differentiable(self):
        torch.manual_seed(0)
        disc = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        loss = alignment._dann_loss(disc, torch.randn(16, 8), torch.randn(16, 8))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class ScheduleTests(unittest.TestCase):
    def test_weights_stay_zero_during_warmup_then_reach_the_maximum(self):
        for fn, w_max in ((schedules.get_mmd_weight, 0.05), (schedules.get_err_weight, 0.2)):
            self.assertEqual(fn(0, w_max=w_max, warmup=5, ramp=10), 0.0)
            self.assertEqual(fn(4, w_max=w_max, warmup=5, ramp=10), 0.0)
            mid = fn(10, w_max=w_max, warmup=5, ramp=10)
            self.assertGreater(mid, 0.0)
            self.assertLess(mid, w_max)
            self.assertAlmostEqual(fn(100, w_max=w_max, warmup=5, ramp=10), w_max)


class SummaryTests(unittest.TestCase):
    def test_jsonify_converts_numpy_scalars_and_arrays(self):
        out = summary._jsonify({"a": np.float32(1.5), "b": np.array([1, 2]), "c": np.int64(3)})
        self.assertEqual(out["a"], 1.5)
        self.assertEqual(out["b"], [1, 2])
        self.assertEqual(out["c"], 3)

    def test_float_or_none_passes_through_nan_as_none(self):
        self.assertIsNone(summary._float_or_none(float("nan")))
        self.assertEqual(summary._float_or_none("2.5"), 2.5)


if __name__ == "__main__":
    unittest.main()

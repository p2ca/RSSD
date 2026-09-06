"""Data-layer checks against a real parsed dataset.

The reservoir tensors are not distributed with this repository, so these tests skip unless
``RSSD_DATA`` points at a directory containing ``parsed/snow_source_v2``:

    RSSD_DATA=/path/to/data python -m unittest tests.test_data_layer -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rssd import paths                                              # noqa: E402

TAG = "snow_source_v2"
HAVE_DATA = paths.parsed_dir(TAG).is_dir()

# Frozen protocol v2: the 13 snow source reservoirs, in node-index order.
SNOW_SOURCE_V2 = ["CO00004", "CO01281", "ND00146", "NE01055", "NE01056", "NE01057",
                  "NE01058", "NE01059", "NE01060", "NE01061", "NE01066", "NE01518", "SD01093"]


@unittest.skipUnless(HAVE_DATA, f"set RSSD_DATA to a directory containing parsed/{TAG}")
class ParsedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rssd.data import datasets
        cls.datasets = datasets
        cls.ds = datasets.load_parsed_dataset(TAG, "local")

    def test_node_count_and_order_match_the_frozen_protocol(self):
        self.assertEqual(self.ds.num_nodes, 13)
        self.assertEqual(self.ds.reservoir_names_in_node_order, SNOW_SOURCE_V2)

    def test_window_and_horizon_match_the_frozen_protocol(self):
        n_windows, t_in, n_nodes, n_features = self.ds.X_train.shape
        self.assertEqual(t_in, 30)
        self.assertEqual(n_nodes, 13)
        self.assertEqual(n_features, 4)
        self.assertEqual(self.ds.pred_len, 7)

    def test_every_reservoir_has_a_local_target_scaler(self):
        local_y = self.ds.scaler_data["local_scalers_y"]
        for name in self.ds.reservoir_names_in_node_order:
            self.assertIn(name, local_y)

    def test_train_scale_statistics_are_positive_and_named(self):
        scale_arr, scale_dict, var_arr, _ = self.datasets.compute_train_scale_stats(self.ds)
        self.assertEqual(len(scale_dict), self.ds.num_nodes)
        self.assertEqual(set(scale_dict), set(self.ds.reservoir_names_in_node_order))
        self.assertTrue((scale_arr > 0).all())
        self.assertTrue((var_arr > 0).all())

    def test_embargo_is_the_input_window_plus_the_horizon(self):
        self.assertEqual(self.datasets.resolve_embargo_windows(self.ds), 30 + 7 - 1)

    def test_source_training_uses_every_training_window(self):
        train, val, test = self.datasets.build_datasets(self.ds)
        embargo = self.datasets.resolve_embargo_windows(self.ds)
        self.assertEqual(len(train), self.ds.X_train.shape[0])
        self.assertEqual(len(test), self.ds.X_test.shape[0] - embargo)
        self.assertGreater(len(val), 0)

    def test_validation_and_test_blocks_are_embargoed_at_the_head(self):
        support, val, test = self.datasets.build_target_datasets(self.ds)
        embargo = self.datasets.resolve_embargo_windows(self.ds)
        self.assertEqual(len(support), self.ds.X_train.shape[0])
        self.assertEqual(len(test), self.ds.X_test.shape[0] - embargo)
        # the validation block is scored on its own windows, never on support windows
        self.assertGreater(len(val), 0)
        self.assertLess(len(val), len(support))


if __name__ == "__main__":
    unittest.main()

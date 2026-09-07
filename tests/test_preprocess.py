"""Preprocessing, checked against the synthetic sample bundle shipped with the code.

These run anywhere: the sample records are part of the repository, so nothing here needs
the reservoir database.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rssd import paths                                                       # noqa: E402
from rssd.data import preprocess                                             # noqa: E402

SAMPLE = paths.SAMPLE_DATA_DIR
SOURCE_POOL = ["SYN0001", "SYN0002", "SYN0003"]
TARGET_POOL = ["SYN0004", "SYN0005"]


class SampleBundleTests(unittest.TestCase):
    """The bundle keeps the column contract the preprocessing step documents."""

    def test_every_pooled_reservoir_has_a_record(self):
        for name in SOURCE_POOL + TARGET_POOL:
            with self.subTest(reservoir=name):
                self.assertTrue((SAMPLE / "align" / f"{name}.csv").is_file())

    def test_records_carry_the_documented_columns(self):
        frame = pd.read_csv(SAMPLE / "align" / "SYN0001.csv")
        for column in ["date"] + preprocess.RECORD_COLUMNS:
            with self.subTest(column=column):
                self.assertIn(column, frame.columns)
        self.assertGreater(len(frame), 30 + 7)

    def test_the_attribute_table_covers_every_reservoir(self):
        table = pd.read_csv(SAMPLE / "meta" / "reservoir_latlon_elev_surface_area.csv")
        for column in ("NIDID", "LATITUDE", "LONGITUDE", "ELEV_M", "SURFACE_AREA_KM2"):
            self.assertIn(column, table.columns)
        self.assertEqual(sorted(table["NIDID"]), sorted(SOURCE_POOL + TARGET_POOL))

    def test_the_pool_lists_match_the_records(self):
        for tag, expected in (("sample_source", SOURCE_POOL), ("sample_target", TARGET_POOL)):
            with self.subTest(pool=tag):
                listed = (SAMPLE / f"reservoirs_{tag}.txt").read_text(encoding="utf-8").split()
                self.assertEqual(listed, expected)


class LeafTests(unittest.TestCase):
    def test_nonpositive_runs_are_filled_from_both_neighbours(self):
        filled = preprocess.fill_nonpositive_runs(np.array([10.0, 0.0, 0.0, 20.0]))
        np.testing.assert_allclose(filled, [10.0, 15.0, 15.0, 20.0])

    def test_a_leading_invalid_run_falls_back_to_the_one_neighbour(self):
        filled = preprocess.fill_nonpositive_runs(np.array([-1.0, -1.0, 8.0]))
        np.testing.assert_allclose(filled, [8.0, 8.0, 8.0])

    def test_truncation_keeps_the_most_recent_years(self):
        frame = pd.DataFrame({"date": pd.date_range("2000-01-01", "2020-12-31", freq="D")})
        kept = preprocess.truncate_recent_history_by_years(frame, years=10)
        self.assertEqual(kept["date"].max(), frame["date"].max())
        self.assertGreaterEqual(kept["date"].min(), pd.Timestamp("2010-12-30"))

    def test_windows_pair_an_input_block_with_the_following_horizon(self):
        data = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
        X, y = preprocess.create_sliding_windows(data, days_x=30, days_y=7, inflow_col_idx=0)
        self.assertEqual(X.shape, (100 - 30 - 7 + 1, 30, 4))
        self.assertEqual(y.shape, (100 - 30 - 7 + 1, 7))
        # the forecast block starts the day after the input block ends
        np.testing.assert_allclose(y[0], data[30:37, 0])

    def test_one_window_starts_on_every_usable_day(self):
        data = np.arange(200 * 4, dtype=np.float32).reshape(200, 4)
        X, _ = preprocess.create_sliding_windows(data, days_x=30, days_y=7, inflow_col_idx=0)
        self.assertEqual(X.shape[0], 200 - 30 - 7 + 1)
        # consecutive windows are one day apart
        np.testing.assert_allclose(X[1, 0], data[1])

    def test_a_record_shorter_than_one_window_yields_nothing(self):
        data = np.zeros((20, 4), dtype=np.float32)
        X, y = preprocess.create_sliding_windows(data, days_x=30, days_y=7, inflow_col_idx=0)
        self.assertEqual(X.shape, (0, 30, 4))
        self.assertEqual(y.shape, (0, 7))

    def test_the_split_is_chronological_and_covers_every_window(self):
        X = np.arange(200 * 2, dtype=np.float32).reshape(200, 2, 1)
        y = np.arange(200, dtype=np.float32).reshape(200, 1)
        blocks = preprocess.split_train_val_test(X, y, train_ratio=0.70, val_ratio=0.15)
        sizes = [blocks[s]["X"].shape[0] for s in ("train", "val", "test")]
        self.assertEqual(sum(sizes), 200)
        self.assertEqual(sizes[0], 140)
        # each block starts where the previous one ended: no shuffling, no overlap
        self.assertEqual(float(blocks["val"]["y"][0, 0]), float(blocks["train"]["y"][-1, 0]) + 1)
        self.assertEqual(float(blocks["test"]["y"][0, 0]), float(blocks["val"]["y"][-1, 0]) + 1)


class ParsedDatasetTests(unittest.TestCase):
    """The whole step, run on the sample source pool."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = preprocess.build_parsed_dataset(
            "sample_source", data_root=SAMPLE, role="source",
            output_dir=Path(cls._tmp.name) / "sample_source")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_it_writes_the_two_files_the_loader_reads(self):
        for name in ("all_rsr_data_local.pkl", "_GNN_supervise_local.pt"):
            with self.subTest(file=name):
                self.assertTrue((Path(self.out) / name).is_file())

    def test_the_recorded_parameters_are_the_frozen_protocol(self):
        with open(Path(self.out) / "all_rsr_data_local.pkl", "rb") as f:
            params = pickle.load(f)["params"]
        self.assertEqual(params["days_x"], 30)
        self.assertEqual(params["days_y"], 7)
        self.assertEqual(params["scaler_type"], "local")
        self.assertEqual(params["feature_cols"], ["inflow", "precip", "tmax", "tmin"])
        self.assertEqual(params["train_ratio"], 0.70)
        self.assertEqual(params["val_ratio"], 0.15)
        self.assertNotIn("y_transform", params)

    def test_every_reservoir_gets_its_own_scaler(self):
        with open(Path(self.out) / "all_rsr_data_local.pkl", "rb") as f:
            blocks = pickle.load(f)
        for name in SOURCE_POOL:
            with self.subTest(reservoir=name):
                self.assertIn(name, blocks["local_scalers_y"])
                self.assertIn(name, blocks)

    def test_the_stacked_tensors_have_the_node_axis_the_model_expects(self):
        sup = torch.load(Path(self.out) / "_GNN_supervise_local.pt",
                         map_location="cpu", weights_only=False)
        X_train = sup["supervised_data"]["X_train"]
        y_train = sup["supervised_data"]["y_train"]
        self.assertEqual(X_train.shape[1:], (30, len(SOURCE_POOL), 4))
        self.assertEqual(y_train.shape[1:], (len(SOURCE_POOL), 7))
        self.assertEqual(sup["graph_data"]["encode_map"],
                         {name: i for i, name in enumerate(SOURCE_POOL)})


if __name__ == "__main__":
    unittest.main()

"""The results figures: NSE classification and how the evaluation tables are read.

These run anywhere — the evaluation tables are written into a temporary directory.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rssd import paths                                                       # noqa: E402
from rssd.config import MODEL_VARIANTS                                       # noqa: E402
from rssd.figures import nse_stability as stability                          # noqa: E402

RESERVOIRS = ["AAA00001", "BBB00002"]


def write_daily_table(run_dir: Path, timestamp: str, values: dict) -> None:
    """One ``per_reservoir_daily_r2_*.csv`` holding a flat NSE per reservoir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict({"reservoir": name}, **{f"r2_d{d}": nse for d in range(1, 8)})
            for name, nse in values.items()]
    pd.DataFrame(rows).to_csv(run_dir / f"per_reservoir_daily_r2_{timestamp}.csv", index=False)


class ClassificationTests(unittest.TestCase):
    """The five classes the manuscript defines, including where the edges fall."""

    def test_each_band_maps_to_its_class(self):
        for value, expected in ((0.80, "Very good"), (0.70, "Good"), (0.60, "Satisfactory"),
                                (0.45, "Acceptable"), (0.10, "Unsatisfactory"),
                                (-2.0, "Unsatisfactory")):
            with self.subTest(nse=value):
                self.assertEqual(stability.classify_nse(value), expected)

    def test_a_boundary_belongs_to_the_lower_class(self):
        for value, expected in ((0.75, "Good"), (0.65, "Satisfactory"),
                                (0.50, "Acceptable"), (0.40, "Unsatisfactory")):
            with self.subTest(nse=value):
                self.assertEqual(stability.classify_nse(value), expected)


class FigureSpecTests(unittest.TestCase):
    def test_every_figure_names_variants_the_package_knows(self):
        for figure, spec in stability.FIGURES.items():
            for variant, _hatch in spec["variants"]:
                with self.subTest(figure=figure, variant=variant):
                    self.assertIn(variant, MODEL_VARIANTS)

    def test_each_panel_pools_two_single_regime_scenarios(self):
        for _title, _pool, scenarios in stability.PANELS:
            self.assertEqual(len(scenarios), 2)
            self.assertNotIn("mixed", " ".join(scenarios))


class EvaluationTableTests(unittest.TestCase):
    """Reading the tables `rssd.cli.evaluate` writes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.logs = Path(self._tmp.name) / "logs"
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(paths, "LOGS_DIR", self.logs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_dir(self, scenario, variant, version="v2"):
        from rssd.config import eval_run_dir
        return Path(eval_run_dir(scenario, variant, version))

    def test_a_missing_run_names_the_command_that_would_produce_it(self):
        with self.assertRaises(FileNotFoundError) as caught:
            stability.resolve_daily_csv("snow2snow", "rssd_lstm", "v2")
        self.assertIn("rssd.cli.evaluate", str(caught.exception))

    def test_the_most_recent_run_is_used_and_a_timestamp_pins_one(self):
        run_dir = self.run_dir("snow2snow", "rssd_lstm")
        write_daily_table(run_dir, "202601010000", {RESERVOIRS[0]: 0.10})
        write_daily_table(run_dir, "202612310000", {RESERVOIRS[0]: 0.90})

        self.assertEqual(
            stability.resolve_daily_csv("snow2snow", "rssd_lstm", "v2").name,
            "per_reservoir_daily_r2_202612310000.csv")
        self.assertEqual(
            stability.resolve_daily_csv("snow2snow", "rssd_lstm", "v2",
                                        timestamp="202601010000").name,
            "per_reservoir_daily_r2_202601010000.csv")

    def test_counts_come_from_the_mean_of_the_two_scenarios(self):
        # 0.30 and 0.90 average to 0.60 -> satisfactory for all seven lead days;
        # 0.80 and 0.80 stay very good.
        write_daily_table(self.run_dir("snow2snow", "rssd_lstm"), "202601010000",
                          {RESERVOIRS[0]: 0.30, RESERVOIRS[1]: 0.80})
        write_daily_table(self.run_dir("rain2snow", "rssd_lstm"), "202601010000",
                          {RESERVOIRS[0]: 0.90, RESERVOIRS[1]: 0.80})

        counts = stability.class_counts("rssd_lstm", ("snow2snow", "rain2snow"),
                                        RESERVOIRS, "v2")
        class_names = [name for name, _colour in stability.NSE_CLASSES]
        self.assertEqual(counts.sum(axis=1).tolist(), [7, 7])
        self.assertEqual(counts[0, class_names.index("Satisfactory")], 7)
        self.assertEqual(counts[1, class_names.index("Very good")], 7)

    def test_a_reservoir_absent_from_a_table_is_an_error(self):
        write_daily_table(self.run_dir("snow2snow", "rssd_lstm"), "202601010000",
                          {RESERVOIRS[0]: 0.5})
        write_daily_table(self.run_dir("rain2snow", "rssd_lstm"), "202601010000",
                          {RESERVOIRS[0]: 0.5})
        with self.assertRaises(KeyError):
            stability.class_counts("rssd_lstm", ("snow2snow", "rain2snow"),
                                   RESERVOIRS, "v2")


if __name__ == "__main__":
    unittest.main()

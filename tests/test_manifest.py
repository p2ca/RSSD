"""The experiment manifest and the quickstart example stay in step with the code."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rssd.config import MODEL_VARIANTS, SCENARIOS, build_train_cfg          # noqa: E402
from rssd.config import load_manifest, manifest_training_runs               # noqa: E402


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()

    def test_declared_variants_are_the_ones_the_package_knows(self):
        declared = [v["key"] for v in self.manifest["variants"]]
        self.assertEqual(sorted(declared), sorted(MODEL_VARIANTS))
        for variant in declared:
            with self.subTest(variant=variant):
                self.assertEqual(MODEL_VARIANTS[variant]["label"],
                                 [v for v in self.manifest["variants"]
                                  if v["key"] == variant][0]["name"])

    def test_declared_scenarios_match_the_transfer_map(self):
        for scenario in self.manifest["scenarios"]:
            with self.subTest(scenario=scenario["key"]):
                self.assertIn(scenario["key"], SCENARIOS)
                mapped = SCENARIOS[scenario["key"]]
                self.assertEqual(mapped["source_domain"], scenario["source"])
                self.assertEqual(mapped["target_domain"], scenario["target"])
                source = self.manifest["sources"][scenario["source"]]
                target = self.manifest["targets"][scenario["target"]]
                self.assertEqual(mapped["source_tag"], source["dataset"])
                self.assertEqual(mapped["eval_tag"], target["dataset"])

    def test_training_runs_cover_every_variant_on_every_source_pool(self):
        runs = manifest_training_runs(self.manifest)
        self.assertEqual(len(runs), len(MODEL_VARIANTS) * len(self.manifest["sources"]))
        for run in runs:
            with self.subTest(run=f"{run['variant']}/{run['dataset']}"):
                cfg = build_train_cfg(variant=run["variant"], dataset_tag=run["dataset"],
                                      version_tag=run["version"])
                self.assertEqual(cfg["experiment"]["DATASET_TAG"], run["dataset"])

    def test_only_the_alignment_baselines_declare_a_target_pool(self):
        for run in manifest_training_runs(self.manifest):
            aligns = MODEL_VARIANTS[run["variant"]]["align_method"] != "none"
            with self.subTest(run=f"{run['variant']}/{run['dataset']}"):
                self.assertEqual(run["align_target"] is not None, aligns)

    def test_protocol_constants_match_the_frozen_setup(self):
        protocol = self.manifest["protocol"]
        self.assertEqual(protocol["input_window_days"], 30)
        self.assertEqual(protocol["forecast_lead_days"], 7)
        self.assertEqual(protocol["target_history_years"], 10)
        self.assertEqual(len(protocol["static_attributes"]), 6)
        self.assertEqual(len(protocol["dynamic_inputs"]), 4)

    def test_every_referenced_reservoir_list_ships_with_the_repository(self):
        referenced = []
        for source in self.manifest["sources"].values():
            referenced += [source[k] for k in ("reservoirs",) if k in source]
        for target in self.manifest["targets"].values():
            referenced.append(target["reservoirs"])
        for partition in self.manifest["partitions"].values():
            referenced += list(partition.values())
        for rel in referenced:
            with self.subTest(file=rel):
                self.assertTrue((REPO / rel).is_file(), f"{rel} is missing from the repository")


class QuickstartTests(unittest.TestCase):
    def test_the_example_runs_without_data(self):
        sys.path.insert(0, str(REPO / "examples"))
        import quickstart

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            quickstart.main()
        text = buffer.getvalue()
        self.assertIn("dynamic state h_dyn", text)
        for variant in MODEL_VARIANTS:
            self.assertIn(variant, text)


if __name__ == "__main__":
    unittest.main()

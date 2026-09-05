"""The experiment ladder: one configuration, eight variants, unchanged run directories."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rssd.config import DEFAULT_TRAIN_CFG, MODEL_VARIANTS, build_train_cfg, train_run_dir  # noqa: E402


def cfg_for(variant, **kwargs):
    kwargs.setdefault("dataset_tag", "snow_source_v2")
    kwargs.setdefault("version_tag", "v2")
    return build_train_cfg(variant=variant, **kwargs)


class LadderTests(unittest.TestCase):
    """Each rung differs from the next by exactly what its name claims."""

    def test_history_only_variant_switches_every_reservoir_component_off(self):
        m = cfg_for("seq2seq_lstm")["model"]
        self.assertFalse(m["use_reservoir_emb"])
        self.assertFalse(m["use_res_static"])
        self.assertFalse(m["use_meta_only_static"])
        self.assertFalse(m["use_darsd"])
        self.assertEqual(m["lcib_k"], 0)

    def test_attribute_informed_variant_adds_attributes_without_identity(self):
        m = cfg_for("attribute_informed_lstm")["model"]
        self.assertTrue(m["use_meta_only_static"])
        self.assertFalse(m["use_reservoir_emb"])
        self.assertFalse(m["use_darsd"])

    def test_identity_informed_variant_adds_identity_without_attributes(self):
        m = cfg_for("identity_informed_lstm")["model"]
        self.assertTrue(m["use_reservoir_emb"])
        self.assertFalse(m["use_res_static"])
        self.assertFalse(m["use_darsd"])

    def test_fully_informed_variant_uses_both_sources_without_a_transfer_mechanism(self):
        cfg = cfg_for("fully_informed_lstm")
        self.assertTrue(cfg["model"]["use_reservoir_emb"])
        self.assertTrue(cfg["model"]["use_res_static"])
        self.assertFalse(cfg["model"]["use_darsd"])
        self.assertEqual(cfg["adaptation"]["ALIGN_METHOD"], "none")

    def test_alignment_baselines_share_the_fully_informed_base(self):
        base = cfg_for("fully_informed_lstm")["model"]
        for variant, method in (("fully_informed_mmd", "mmd"),
                                ("fully_informed_coral", "coral"),
                                ("fully_informed_dann", "dann")):
            with self.subTest(variant=variant):
                cfg = cfg_for(variant)
                self.assertEqual(cfg["adaptation"]["ALIGN_METHOD"], method)
                self.assertEqual(cfg["adaptation"]["USE_MMD"], method == "mmd")
                self.assertFalse(cfg["model"]["use_darsd"])
                for key in ("use_reservoir_emb", "use_res_static", "latent_mode"):
                    self.assertEqual(cfg["model"][key], base[key])

    def test_rssd_variant_replaces_alignment_with_the_decomposition(self):
        cfg = cfg_for("rssd_lstm")
        self.assertTrue(cfg["model"]["use_darsd"])
        self.assertEqual(cfg["model"]["lcib_k"], 8)
        self.assertEqual(cfg["model"]["darsd_mode"], "softmax_reconstruction")
        self.assertEqual(cfg["adaptation"]["ALIGN_METHOD"], "none")

    def test_transformer_variant_changes_only_the_backbone(self):
        lstm, transformer = cfg_for("rssd_lstm")["model"], cfg_for("rssd_transformer")["model"]
        self.assertEqual(transformer.get("BACKBONE"), "transformer_seq2seq")
        for key in ("use_reservoir_emb", "use_res_static", "use_darsd", "lcib_k", "darsd_mode"):
            self.assertEqual(transformer[key], lstm[key])


class RunDirectoryTests(unittest.TestCase):
    def test_run_directories_keep_the_project_naming(self):
        expected = {
            "seq2seq_lstm": "exp1_pure_lstm_pure_lstm_DARSD0_ERR0",
            "rssd_lstm": "exp2_full_model_full_model_DARSD1_ERR1",
            "fully_informed_mmd": "exp3_full_model_mmd_full_model_DARSD0_ERR1",
            "fully_informed_dann": "exp9_full_model_dann_full_model_DARSD0_ERR1",
            "rssd_transformer": "exp2t_full_model_transformer_full_model_DARSD1_ERR1",
        }
        for variant, group in expected.items():
            with self.subTest(variant=variant):
                path = train_run_dir(cfg_for(variant))
                self.assertEqual(path.name, "v2")
                self.assertEqual(path.parent.name, group)
                self.assertEqual(path.parent.parent.name, "train_snow")

    def test_every_declared_variant_resolves(self):
        for variant in MODEL_VARIANTS:
            with self.subTest(variant=variant):
                self.assertTrue(train_run_dir(cfg_for(variant)).parent.name)


class OverrideTests(unittest.TestCase):
    def test_overrides_apply_and_defaults_are_left_untouched(self):
        cfg = cfg_for("rssd_lstm", overrides={"data": {"BATCH_SIZE": 64}})
        self.assertEqual(cfg["data"]["BATCH_SIZE"], 64)
        self.assertEqual(DEFAULT_TRAIN_CFG["data"]["BATCH_SIZE"], 128)

    def test_unknown_override_keys_are_rejected(self):
        with self.assertRaises(KeyError):
            cfg_for("rssd_lstm", overrides={"data": {"BATCH_SIZE_TYPO": 64}})
        with self.assertRaises(KeyError):
            cfg_for("rssd_lstm", overrides={"nonexistent_section": {"x": 1}})

    def test_validation_fraction_is_the_frozen_value(self):
        self.assertEqual(cfg_for("rssd_lstm")["data"]["VAL_FRAC"], 0.10)
        self.assertEqual(cfg_for("rssd_lstm", overrides={"data": {"VAL_FRAC": 0.2}})
                         ["data"]["VAL_FRAC"], 0.2)


if __name__ == "__main__":
    unittest.main()

"""Experiment identity and evaluation settings.

Two things live here:

* the frozen mapping between a **transfer scenario** (``snow2rain``) and the dataset tags
  it evaluates, and between a **model variant** (``rssd_lstm``) and the checkpoint directory
  it was trained into;
* :class:`EvalConfig`, the evaluation settings.

Directory names keep their original internal identifiers (``exp2_full_model``,
``DARSD1_ERR1``, ...) so that runs produced by this package land beside the existing ones
and every downstream script keeps working. Manuscript-facing names are carried alongside in
:data:`MODEL_VARIANTS` for reporting.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from rssd import paths
from rssd.profiles import (_apply_train_ablation_profile, _apply_train_exp_overrides,
                          _validate_ablation_and_adaptation)

__all__ = ["SCENARIOS", "MODEL_VARIANTS", "run_group", "checkpoint_path", "eval_run_dir",
           "EvalConfig", "DEFAULT_TRAIN_CFG", "build_train_cfg", "train_run_dir",
           "MANIFEST_PATH", "load_manifest", "manifest_training_runs"]


# --------------------------------------------------------------------------- scenarios
# Protocol v2. Sources train the model, targets are held out and adapted on their own
# recent history. "mixed" trains on the pooled snow+rain source reservoirs.
SCENARIOS = {
    "snow2snow": dict(source_tag="snow_source_v2", eval_tag="snow_target_v2",
                      source_domain="snow", target_domain="snow"),
    "rain2rain": dict(source_tag="rain_source", eval_tag="rain_target_v2",
                      source_domain="rain", target_domain="rain"),
    "snow2rain": dict(source_tag="snow_source_v2", eval_tag="rain_target_v2",
                      source_domain="snow", target_domain="rain"),
    "rain2snow": dict(source_tag="rain_source", eval_tag="snow_target_v2",
                      source_domain="rain", target_domain="snow"),
    "mixed2snow": dict(source_tag="mixed_source_v2", eval_tag="snow_target_v2",
                       source_domain="mixed", target_domain="snow"),
    "mixed2rain": dict(source_tag="mixed_source_v2", eval_tag="rain_target_v2",
                       source_domain="mixed", target_domain="rain"),
    # transfer between the two synthetic pools of the sample bundle; present so the
    # pipeline can be exercised end to end without the reservoir records.
    "sample2sample": dict(source_tag="sample_source", eval_tag="sample_target",
                          source_domain="sample", target_domain="sample"),
}


# ---------------------------------------------------------------------- model variants
# key            : stable identifier used on the command line
# exp_name       : internal experiment id, kept for checkpoint/log path compatibility
# model_variant  : architecture family recorded in the checkpoint directory name
# darsd / err    : the frozen DARSD{0,1}_ERR{0,1} suffix of the directory name
# align_method   : none | mmd | coral | dann
# label          : manuscript-facing name
MODEL_VARIANTS = {
    "seq2seq_lstm": dict(exp_name="exp1_pure_lstm", model_variant="pure_lstm",
                         darsd=0, err=0, align_method="none",
                         label="LSTM baseline"),
    "attribute_informed_lstm": dict(exp_name="exp4_meta_pure_lstm", model_variant="pure_lstm",
                                    darsd=0, err=0, align_method="none",
                                    label="LSTM + attributes"),
    "identity_informed_lstm": dict(exp_name="exp6_context_lstm", model_variant="full_model",
                                   darsd=0, err=0, align_method="none",
                                   label="LSTM + reservoir-ID embedding"),
    "fully_informed_lstm": dict(exp_name="exp7_metacontext_lstm", model_variant="full_model",
                                darsd=0, err=0, align_method="none",
                                label="LSTM + attributes + reservoir-ID embedding"),
    "fully_informed_mmd": dict(exp_name="exp3_full_model_mmd", model_variant="full_model",
                               darsd=0, err=1, align_method="mmd",
                               label="LSTM + attributes + reservoir-ID embedding + MMD"),
    "fully_informed_coral": dict(exp_name="exp8_full_model_coral", model_variant="full_model",
                                 darsd=0, err=1, align_method="coral",
                                 label="LSTM + attributes + reservoir-ID embedding + CORAL"),
    "fully_informed_dann": dict(exp_name="exp9_full_model_dann", model_variant="full_model",
                                darsd=0, err=1, align_method="dann",
                                label="LSTM + attributes + reservoir-ID embedding + DANN"),
    "rssd_lstm": dict(exp_name="exp2_full_model", model_variant="full_model",
                      darsd=1, err=1, align_method="none",
                      label="RSSD (LSTM backbone)"),
    "rssd_transformer": dict(exp_name="exp2t_full_model_transformer", model_variant="full_model",
                             darsd=1, err=1, align_method="none",
                             label="RSSD (Transformer backbone)"),
}


def _variant_spec(variant: str) -> dict:
    if variant in MODEL_VARIANTS:
        return MODEL_VARIANTS[variant]
    for spec in MODEL_VARIANTS.values():           # allow the internal exp id as well
        if spec["exp_name"] == variant:
            return spec
    raise KeyError(f"Unknown model variant {variant!r}. "
                   f"Known: {sorted(MODEL_VARIANTS)}")


def run_group(variant: str) -> str:
    """Checkpoint/log directory name, e.g. ``exp2_full_model_full_model_DARSD1_ERR1``."""
    spec = _variant_spec(variant)
    return f"{spec['exp_name']}_{spec['model_variant']}_DARSD{spec['darsd']}_ERR{spec['err']}"


def checkpoint_path(variant: str, source_domain: str, version: str = "v2",
                    bundle: str = "best_bundle.pt"):
    """``logs/train_<domain>/<run_group>/<version>/best_bundle.pt``."""
    return paths.train_log_dir(source_domain, run_group(variant), version) / bundle


def eval_run_dir(scenario: str, variant: str, version: str = "v2"):
    """``logs/eval_<target domain>/eval_<scenario>_<exp_name>/<version>``."""
    spec = _variant_spec(variant)
    target_domain = SCENARIOS[scenario]["target_domain"]
    return paths.eval_log_dir(target_domain, f"eval_{scenario}_{spec['exp_name']}", version)


# ------------------------------------------------------------------------- eval config
@dataclass
class EvalConfig:
    """Evaluation settings. Defaults are the frozen protocol-v2 values."""

    scaler_type: str = "local"
    seed: int = 42

    # prediction contract
    clamp_pred_to_fr: bool = True

    # target-history adaptation
    enable_finetune: bool = True
    finetune_only_if_domain_shift: bool = True
    finetune_mode: str = "full"
    finetune_max_epochs: int = 80
    finetune_patience: int = 10
    finetune_min_delta: float = 1e-5
    finetune_lr: float = 2e-5
    finetune_weight_decay: float = 5e-4
    finetune_grad_clip: float = 1.0
    finetune_seed: int = 20260312
    finetune_cache_to_device: bool = True
    finetune_train_max_batches: int = None
    finetune_val_max_batches: int = None

    # metric reporting
    overall_r2_mode: str = "micro"          # micro = pooled, macro = reservoir-averaged
    monitor_reservoirs: tuple = ()

    # loaders
    eval_batch_size: int = 128
    eval_num_workers: int = 0
    eval_pin_memory: bool = None

    # unseen target node set
    use_meta_emb_init: bool = True
    meta_emb_ridge_lambda: float = 1e-3
    meta_emb_blend_alpha: float = 0.5
    cross_static_norm_clip: float = 3.0
    cross_static_fatal_p95: float = 0.0

    exclude_reservoirs: set = field(default_factory=set)

    def as_metadata(self) -> dict:
        """Flat dictionary of the settings, for the run's metadata record."""
        return {k: (sorted(v) if isinstance(v, set) else v) for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- training
# The frozen training configuration. It is
# kept as the same nested dictionary because a checkpoint stores it verbatim under
# ``user_cfg`` and evaluation reads fields back out of it; a parallel schema would drift.
# ``build_train_cfg`` below is the only supported way to derive a run's configuration:
# it applies the variant profile, the experiment overrides and the contract checks in the
# order the training run does.
DEFAULT_TRAIN_CFG = {
    "experiment": {
        "VERSION_TAG": "v1",
        "DATASET_TAG": "rain_source",
        "SCALER_TYPE": "local",
        "RUN_TAG": "AUTO",
        "OVERALL_R2_MODE": "micro",
        "SEED": 42,
    },

    "model": {
        "use_reservoir_emb": True,
        "res_emb_dim": 8,
        "emb_dropout_p": 0.3,  # zero out embedding vector with prob 0.3 during training for cold-start robustness

        "use_res_static": True,
        "res_static_dim": 6,

        "use_meta_only_static": False,
        "meta_only_static_dim": 0,
        "meta_feature_names": ["storage_max", "elev_mean", "surface_area", "lat", "lon", "ground_elev"],

        "latent_mode": "attn",          # last / attn
        "use_latent_proj": True,

        "use_darsd": True,
        "lcib_k": 16,

        "HIDDEN_DIM": 256,
        "NUM_LAYERS": 1,
        "DROPOUT": 0.15,
    },

    "objective": {
        "CLAMP_PRED_TO_FR": True,
        "DARSD_WEIGHT": 0.0002,
    },

    "train": {
        "LR": 2e-4,
        "WEIGHT_DECAY": 2e-4,
        "GRAD_CLIP_NORM": 0.5,

        "MAX_EPOCHS": 200,
        "EARLY_STOP_PATIENCE": 20,
        "MIN_DELTA": 1e-3,
        "MIN_DELTA_REL": 0.005,
        "EARLY_STOP_MIN_EPOCHS": 25,

        "STOP_ON_MIN_LR": True,
        "STOP_ON_MIN_LR_PATIENCE": 4,

        "USE_CHECKPOINT_AVG": True,
        "CHECKPOINT_AVG_LAST_K": 5,
    },

    "data": {
        "BATCH_SIZE": 128,



        "DL_NUM_WORKERS": 2,
        "DL_PIN_MEMORY": None,
        "DL_PREFETCH_FACTOR": 2,
    },
    
    "ablation": {
        "EXP_NAME": "exp2_full_model",     # exp1_pure_lstm | exp2_full_model | exp3_full_model_mmd | exp4_meta_pure_lstm
        "MODEL_VARIANT": "full_model",     # pure_lstm | full_model
    },

    "adaptation": {
        "TARGET_DATASET_TAG": "rain_target",         # e.g. snow_target / rain_target ; used when ALIGN_METHOD != "none"
        "ALIGN_METHOD": "none",                      # none | mmd | coral | dann  (domain-alignment master switch)
        "USE_MMD": False,                            # legacy alias; derived from ALIGN_METHOD=="mmd"

        "MMD_WEIGHT_MAX": 0.01,
        "MMD_WARMUP_EPOCHS": 8,
        "MMD_RAMP_EPOCHS": 12,

        "MMD_KERNEL_NUM": 5,
        "MMD_KERNEL_MUL": 2.0,
        "MMD_NORMALIZE_LATENT": True,
        "MMD_MAX_SAMPLES_PER_DOMAIN": 256,

        # CORAL (exp8): weight on deep-CORAL covariance-distance loss
        "CORAL_WEIGHT_MAX": 1.0,
        # DANN (exp9): GRL lambda ceiling + domain-discriminator config
        "DANN_LAMBDA_MAX": 1.0,
        "DANN_DISC_HIDDEN": 128,
        "DANN_DISC_LR": 2e-4,
    },
}


def build_train_cfg(variant: str = None, dataset_tag: str = None, version_tag: str = None,
                    seed: int = None, target_dataset_tag: str = None, align_method: str = None,
                    overrides: dict = None) -> dict:
    """Derive one run's configuration from the defaults.

    ``variant`` accepts either a key of :data:`MODEL_VARIANTS` (``rssd_lstm``) or an internal
    experiment id (``exp2_full_model``); both set ``EXP_NAME`` and ``MODEL_VARIANT``.
    ``overrides`` is a nested dictionary merged in before the profiles are applied.
    """
    cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)

    if variant is not None:
        spec = _variant_spec(variant)
        cfg["ablation"]["EXP_NAME"] = spec["exp_name"]
        cfg["ablation"]["MODEL_VARIANT"] = spec["model_variant"]
        cfg["adaptation"]["ALIGN_METHOD"] = spec["align_method"]
    if dataset_tag is not None:
        cfg["experiment"]["DATASET_TAG"] = str(dataset_tag)
    if version_tag is not None:
        cfg["experiment"]["VERSION_TAG"] = str(version_tag)
    if seed is not None:
        cfg["experiment"]["SEED"] = int(seed)
    if target_dataset_tag is not None:
        cfg["adaptation"]["TARGET_DATASET_TAG"] = str(target_dataset_tag)
    if align_method is not None:
        cfg["adaptation"]["ALIGN_METHOD"] = str(align_method)

    for section, values in (overrides or {}).items():
        if section not in cfg:
            raise KeyError(f"unknown configuration section {section!r}; "
                           f"known: {sorted(cfg)}")
        if not isinstance(values, dict):
            raise TypeError(f"overrides[{section!r}] must be a dictionary")
        unknown = [k for k in values if k not in cfg[section]]
        if unknown:
            raise KeyError(f"unknown keys in {section!r}: {unknown}")
        cfg[section].update(values)

    cfg["adaptation"]["USE_MMD"] = cfg["adaptation"]["ALIGN_METHOD"] == "mmd"

    cfg = _apply_train_ablation_profile(cfg, cfg["ablation"]["MODEL_VARIANT"])
    cfg = _apply_train_exp_overrides(cfg)
    _validate_ablation_and_adaptation(cfg)
    return cfg


def train_run_dir(cfg: dict):
    """``logs/train_<domain>/<exp>_<variant>_DARSD{0,1}_ERR{0,1}/<version>``."""
    domain = str(cfg["experiment"]["DATASET_TAG"]).split("_")[0]
    spec = _variant_spec(str(cfg["ablation"]["EXP_NAME"]))
    group = (f"{cfg['ablation']['EXP_NAME']}_{cfg['ablation']['MODEL_VARIANT']}"
             f"_DARSD{int(bool(cfg['model']['use_darsd']))}_ERR{spec['err']}")
    return paths.train_log_dir(domain, group, str(cfg["experiment"]["VERSION_TAG"]))


# --------------------------------------------------------------------------- manifest
# The manifest ships with the code, so it is located relative to this file rather than to
# RSSD_ROOT (which points at a working copy of the data).
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "experiments.yaml"


def load_manifest(path=None) -> dict:
    """Read ``configs/experiments.yaml``: which variants were trained on which sources."""
    import yaml                                    # optional; only the manifest needs it

    with open(path or MANIFEST_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def manifest_training_runs(manifest: dict = None) -> list:
    """Every (variant, source pool) training run the manifest declares."""
    manifest = manifest or load_manifest()
    version = str(manifest.get("version", "v2"))
    runs = []
    for variant in manifest["variants"]:
        for pool, source in manifest["sources"].items():
            spec = _variant_spec(variant["key"])
            runs.append({
                "variant": variant["key"],
                "name": variant["name"],
                "pool": pool,
                "dataset": source["dataset"],
                "version": version,
                "align_target": (source.get("align_target")
                                 if spec["align_method"] != "none" else None),
            })
    return runs

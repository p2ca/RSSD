"""Experiment profiles: turning one training configuration into the eight variants.

A run states which variant it is
(``MODEL_VARIANT``: the architecture family) and which experiment
(``EXP_NAME``: the specific ablation), and these functions rewrite the configuration
accordingly -- switching the per-reservoir embedding, static-attribute conditioning, the
error head and the RSSD layer on or off, and selecting the alignment objective. Keeping
this as one auditable transform is what makes the ladder of variants a controlled
comparison rather than eight hand-maintained configurations.
"""

from __future__ import annotations

import copy

__all__ = ["_resolve_align_method", "_apply_train_ablation_profile",
           "_apply_train_exp_overrides", "_validate_ablation_and_adaptation"]


def _resolve_align_method(ad: dict) -> str:
    """Master alignment switch. ALIGN_METHOD in {none,mmd,coral,dann};
    legacy USE_MMD=True maps to 'mmd' when ALIGN_METHOD is unset/none."""
    am = str(ad.get("ALIGN_METHOD", "none")).strip().lower()
    if am in ("", "none") and bool(ad.get("USE_MMD", False)):
        am = "mmd"
    if am not in ("none", "mmd", "coral", "dann"):
        raise ValueError(f"Invalid ALIGN_METHOD={am!r}; allowed: none|mmd|coral|dann")
    return am


def _apply_train_ablation_profile(cfg: dict, model_variant: str) -> dict:
    cfg = copy.deepcopy(cfg)
    m = cfg["model"]

    if model_variant == "pure_lstm":
        # history-only: keep the plain sequence-to-sequence LSTM trunk
        m["USE_DIRECT_HEAD"] = False

        m["use_reservoir_emb"] = False
        m["res_emb_dim"] = 0

        m["use_res_static"] = False
        m["res_static_dim"] = 0

        m["latent_mode"] = "last"
        m["use_latent_proj"] = False

        m["use_err_head"] = False
        m["err_head_hidden"] = 0

        m["use_darsd"] = False
        m["lcib_k"] = 0

        m["use_meta_only_static"] = False
        m["meta_only_static_dim"] = 0
        m.setdefault("meta_feature_names", ["storage_max", "elev_mean", "lat", "lon", "ground_elev"])
        m.setdefault("meta_feature_strengths", [1.0, 1.0, 1.0, 1.0, 1.0])

    elif model_variant == "full_model":
        # full model: every reservoir-information component enabled
        m["USE_DIRECT_HEAD"] = True

        m["use_reservoir_emb"] = True
        m["res_emb_dim"] = 4

        m["use_res_static"] = True
        m["res_static_dim"] = 6
        m["RES_STATIC_MODE"] = "latent"
        m["FILM_GAMMA_SCALE"] = 0.0   # legacy field, no longer used by runtime path
        m["FILM_BETA_SCALE"] = 0.0    # legacy field, no longer used by runtime path
        m["FILM_GAMMA_SCALE"] = 0.10
        m["FILM_BETA_SCALE"] = 0.10

        m["latent_mode"] = "attn"
        m["use_latent_proj"] = True

        m["use_err_head"] = True
        m["err_head_hidden"] = 64

        m["use_darsd"] = True
        m["lcib_k"] = 8

        m["use_meta_only_static"] = False
        m["meta_only_static_dim"] = 0
        m.setdefault("meta_feature_names", ["storage_max", "elev_mean", "lat", "lon", "ground_elev"])
        m.setdefault("meta_feature_strengths", [1.0, 1.0, 1.0, 1.0, 1.0])

    return cfg


def _apply_train_exp_overrides(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    exp_name = str(cfg["ablation"]["EXP_NAME"]).strip()
    m = cfg["model"]

    if exp_name == "exp3_full_model_mmd":
        # keep full-model backbone, but remove DARSD only
        m["use_darsd"] = False
        m["lcib_k"] = 0

    elif exp_name == "exp4_meta_pure_lstm":
        # pure_lstm backbone + metadata only
        m["use_meta_only_static"] = True
        m["meta_only_static_dim"] = 6

        if "meta_feature_names" not in m:
            m["meta_feature_names"] = ["storage_max", "elev_mean", "lat", "lon", "ground_elev"]
        if "meta_feature_strengths" not in m:
            m["meta_feature_strengths"] = [1.0] * int(m["meta_only_static_dim"])

        if len(m["meta_feature_names"]) != int(m["meta_only_static_dim"]):
            raise ValueError(
                f"exp4_meta_pure_lstm requires len(meta_feature_names)==meta_only_static_dim, "
                f"got {len(m['meta_feature_names'])} vs {m['meta_only_static_dim']}"
            )
        if len(m["meta_feature_strengths"]) != int(m["meta_only_static_dim"]):
            raise ValueError(
                f"exp4_meta_pure_lstm requires len(meta_feature_strengths)==meta_only_static_dim, "
                f"got {len(m['meta_feature_strengths'])} vs {m['meta_only_static_dim']}"
            )

        tr = cfg["train"]
        tr["MIN_DELTA"] = 1e-4
        tr["MIN_DELTA_REL"] = 0.0
        tr["EARLY_STOP_MIN_EPOCHS"] = max(int(tr["EARLY_STOP_MIN_EPOCHS"]), 25)
        tr["EARLY_STOP_PATIENCE"] = max(int(tr["EARLY_STOP_PATIENCE"]), 20)

    elif exp_name == "exp5_darsd_pure_lstm":
        # pure_lstm backbone + metadata + DARSD (no reservoir embedding)
        m["use_meta_only_static"] = True
        m["meta_only_static_dim"] = 6
        if "meta_feature_names" not in m:
            m["meta_feature_names"] = ["storage_max", "elev_mean", "lat", "lon", "ground_elev"]
        if "meta_feature_strengths" not in m:
            m["meta_feature_strengths"] = [1.0] * 6
        m["use_darsd"] = True
        m["lcib_k"] = 16

        tr = cfg["train"]
        tr["MIN_DELTA"] = 1e-4
        tr["MIN_DELTA_REL"] = 0.0
        tr["EARLY_STOP_MIN_EPOCHS"] = max(int(tr["EARLY_STOP_MIN_EPOCHS"]), 25)
        tr["EARLY_STOP_PATIENCE"] = max(int(tr["EARLY_STOP_PATIENCE"]), 20)

    elif exp_name == "exp6_context_lstm":
        # full_model backbone + reservoir embedding only; no metadata, no DARSD, no err_head
        m["use_res_static"] = False
        m["res_static_dim"] = 0
        m["use_darsd"] = False
        m["lcib_k"] = 0
        m["use_err_head"] = False
        m["err_head_hidden"] = 0

    elif exp_name == "exp7_metacontext_lstm":
        # full_model backbone + reservoir embedding + static metadata; no DARSD, no err_head
        m["use_darsd"] = False
        m["lcib_k"] = 0
        m["use_err_head"] = False
        m["err_head_hidden"] = 0

    elif exp_name == "exp2t_full_model_transformer":
        # backbone sensitivity: retain the information/RSSD design while replacing
        # the complete LSTM forecasting backbone with a Transformer encoder-decoder.
        m["BACKBONE"]    = "transformer_seq2seq"
        m["N_HEADS"]     = 8
        m["TF_LAYERS"]   = 2
        m["TF_FF_MULT"]  = 4
        m["TIN"]         = 30

    elif exp_name in ("exp8_full_model_coral", "exp9_full_model_dann"):
        # alignment baselines: identical to exp3 (full_model, DARSD removed);
        # only the domain-alignment loss differs (CORAL / DANN instead of MMD).
        m["use_darsd"] = False
        m["lcib_k"] = 0

    return cfg


def _validate_ablation_and_adaptation(cfg: dict) -> None:
    exp_name = str(cfg["ablation"]["EXP_NAME"]).strip()
    model_variant = str(cfg["ablation"]["MODEL_VARIANT"]).strip().lower()
    ad = cfg["adaptation"]
    align_method = _resolve_align_method(ad)
    use_align = align_method in ("mmd", "coral", "dann")
    dataset_tag = str(cfg["experiment"]["DATASET_TAG"]).strip()
    target_dataset_tag = str(ad["TARGET_DATASET_TAG"]).strip()

    def _require_source_and_target():
        if not (dataset_tag.endswith("_source") or dataset_tag.endswith("_source_v2")):
            raise ValueError(f"alignment train must use *_source or *_source_v2 dataset tag, got {dataset_tag}")
        if (not target_dataset_tag) or not (target_dataset_tag.endswith("_target") or target_dataset_tag.endswith("_target_v2")):
            raise ValueError(f"alignment train must specify *_target or *_target_v2 TARGET_DATASET_TAG, got {target_dataset_tag}")

    # exp -> required align method (all alignment baselines are full_model, DARSD off)
    _ALIGN_EXPS = {
        "exp3_full_model_mmd":   "mmd",
        "exp8_full_model_coral": "coral",
        "exp9_full_model_dann":  "dann",
    }
    _NOALIGN_FULL = {"exp2_full_model", "exp6_context_lstm", "exp7_metacontext_lstm",
                     "exp2t_full_model_transformer"}
    _NOALIGN_PURE = {"exp1_pure_lstm", "exp4_meta_pure_lstm", "exp5_darsd_pure_lstm"}

    if exp_name in _ALIGN_EXPS:
        if model_variant != "full_model":
            raise ValueError(f"{exp_name} requires MODEL_VARIANT='full_model'")
        if align_method != _ALIGN_EXPS[exp_name]:
            raise ValueError(f"{exp_name} requires ALIGN_METHOD='{_ALIGN_EXPS[exp_name]}', got '{align_method}'")
        _require_source_and_target()
    elif exp_name in _NOALIGN_FULL:
        if model_variant != "full_model":
            raise ValueError(f"{exp_name} requires MODEL_VARIANT='full_model'")
        if use_align:
            raise ValueError(f"{exp_name} must not enable domain alignment (ALIGN_METHOD={align_method})")
    elif exp_name in _NOALIGN_PURE:
        if model_variant != "pure_lstm":
            raise ValueError(f"{exp_name} requires MODEL_VARIANT='pure_lstm'")
        if use_align:
            raise ValueError(f"{exp_name} must not enable domain alignment (ALIGN_METHOD={align_method})")
    else:
        raise ValueError(f"Unsupported EXP_NAME={exp_name}")

    _valid_model_variants = {"pure_lstm", "full_model"}
    if model_variant not in _valid_model_variants:
        raise ValueError(f"Invalid MODEL_VARIANT={model_variant}, allowed={sorted(_valid_model_variants)}")

"""Checkpoint-locked model configuration.

Evaluation never re-specifies the architecture: it reads it back from the training bundle,
so an evaluation run cannot silently disagree with the weights it loads. This module parses
that stored configuration and enforces the contract checks it has to satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rssd.data.static_attrs import LEGACY_STATIC_ATTRIBUTE_NAMES

__all__ = ["CheckpointConfig", "load_checkpoint_config", "VALID_LATENT_MODES"]

VALID_LATENT_MODES = {"last", "attn"}
SUPPORTED_BACKBONES = {"lstm", "transformer_seq2seq"}

REQUIRED_CONFIG_KEYS = [
    "hidden_dim", "num_layers", "dropout",
    "use_reservoir_emb", "reservoir_emb_dim",
    "use_res_static", "res_static_dim",
    "latent_mode", "use_latent_proj",
    "use_darsd", "lcib_k",
]


@dataclass
class CheckpointConfig:
    """Architecture and provenance of a trained bundle."""

    # architecture
    hidden_dim: int
    num_layers: int
    dropout: float
    latent_mode: str
    use_latent_proj: bool

    # backbone (bundles predating the Transformer work carry no backbone key)
    backbone: str = "lstm"
    n_heads: int = 8
    tf_layers: int = 2
    tf_ff_mult: int = 4
    tin: int = 30

    # reservoir identity
    use_reservoir_emb: bool = False
    reservoir_emb_dim: int = 0
    emb_dropout_p: float = 0.0

    # static attributes
    use_res_static: bool = False
    res_static_dim: int = 0
    use_meta_only_static: bool = False
    meta_only_static_dim: int = 0
    meta_feature_names: list = field(default_factory=lambda: list(LEGACY_STATIC_ATTRIBUTE_NAMES))

    # auxiliary heads and RSSD
    use_darsd: bool = False
    lcib_k: int = 0

    # provenance
    dataset_tag: str = ""
    scaler_type: str = ""
    exp_name: str = "exp_unknown"
    model_variant: str = "variant_unknown"
    train_reservoir_names_in_node_order: list = field(default_factory=list)

    def validate(self) -> None:
        """Enforce the architecture contracts the project relies on."""
        if self.latent_mode not in VALID_LATENT_MODES:
            raise ValueError(f"[CKPT] Invalid latent_mode={self.latent_mode}; "
                             f"allowed={sorted(VALID_LATENT_MODES)}")

        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"[CKPT] Unsupported backbone={self.backbone!r}; this package "
                             f"builds {sorted(SUPPORTED_BACKBONES)}.")

        if self.use_res_static and not self.use_reservoir_emb:
            raise ValueError("[CKPT] Contract violation: use_res_static=True requires "
                             "use_reservoir_emb=True")

        n_features = len(self.meta_feature_names)
        if self.use_res_static and int(self.res_static_dim) != n_features:
            raise ValueError(f"[CKPT] Contract violation: use_res_static=True requires "
                             f"res_static_dim==len(meta_feature_names)={n_features}, "
                             f"got {self.res_static_dim}")

        if self.use_meta_only_static and self.use_reservoir_emb:
            raise ValueError("[CKPT] Contract violation: use_meta_only_static=True requires "
                             "use_reservoir_emb=False")
        if self.use_meta_only_static and self.use_res_static:
            raise ValueError("[CKPT] Contract violation: use_meta_only_static=True requires "
                             "use_res_static=False")
        if self.use_meta_only_static and int(self.meta_only_static_dim) != n_features:
            raise ValueError(f"[CKPT] Contract violation: use_meta_only_static=True requires "
                             f"meta_only_static_dim==len(meta_feature_names)={n_features}, "
                             f"got {self.meta_only_static_dim}")

    def describe(self) -> str:
        return (f"exp={self.exp_name} variant={self.model_variant} backbone={self.backbone} "
                f"hidden={self.hidden_dim} emb={self.reservoir_emb_dim} "
                f"static={self.res_static_dim} lcib_k={self.lcib_k} "
                f"dataset={self.dataset_tag}/{self.scaler_type}")


def load_checkpoint_config(ckpt_path, expected_dataset_tag=None, expected_scaler_type=None):
    """Read the architecture back out of a training bundle and check its contracts.

    Parameters
    ----------
    ckpt_path
        Path to ``best_bundle.pt`` / ``avg_bundle.pt``.
    expected_dataset_tag, expected_scaler_type
        When given, the bundle must have been trained with exactly these, otherwise the
        evaluation would silently mix protocols.
    """
    bundle = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(bundle, dict):
        raise TypeError(f"[CKPT] Expected dict-like train bundle, got {type(bundle)} from {ckpt_path}")

    cfg = bundle.get("config", None)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        raise RuntimeError(f"[CKPT] Missing non-empty 'config' in {ckpt_path}. "
                           "Eval is configured to be strict ckpt-locked.")

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"[CKPT] Incomplete config in {ckpt_path}; missing keys: {missing}")

    meta_feature_names = list(cfg.get("meta_feature_names", LEGACY_STATIC_ATTRIBUTE_NAMES))
    meta_only_static_dim = int(cfg.get("meta_only_static_dim", 0))

    for key in ("dataset_tag", "scaler_type"):
        if bundle.get(key, None) is None:
            raise KeyError(f"[CKPT] Missing required metadata key: {key} in {ckpt_path}")

    train_order = bundle.get("train_reservoir_names_in_node_order", None)
    if train_order is None:
        raise KeyError(f"[CKPT] Missing train_reservoir_names_in_node_order in {ckpt_path}. "
                       "Cannot decide whether eval is exact replay or true cross-dataset.")

    config = CheckpointConfig(
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        latent_mode=str(cfg["latent_mode"]),
        use_latent_proj=bool(cfg["use_latent_proj"]),
        backbone=str(cfg.get("backbone", "lstm")),
        n_heads=int(cfg.get("n_heads", 8)),
        tf_layers=int(cfg.get("tf_layers", 2)),
        tf_ff_mult=int(cfg.get("tf_ff_mult", 4)),
        tin=int(cfg.get("tin", 30)),
        use_reservoir_emb=bool(cfg["use_reservoir_emb"]),
        reservoir_emb_dim=int(cfg["reservoir_emb_dim"]),
        emb_dropout_p=float(cfg.get("emb_dropout_p", 0.0)),
        use_res_static=bool(cfg["use_res_static"]),
        res_static_dim=int(cfg["res_static_dim"]),
        use_meta_only_static=bool(cfg.get("use_meta_only_static", False)),
        meta_only_static_dim=meta_only_static_dim,
        meta_feature_names=meta_feature_names,
        use_darsd=bool(cfg["use_darsd"]),
        lcib_k=int(cfg["lcib_k"]),
        dataset_tag=str(bundle["dataset_tag"]),
        scaler_type=str(bundle["scaler_type"]),
        exp_name=str(bundle.get("exp_name", "exp_unknown")),
        model_variant=str(bundle.get("model_variant", "variant_unknown")),
        train_reservoir_names_in_node_order=list(train_order),
    )
    config.validate()

    if expected_dataset_tag is not None and config.dataset_tag != str(expected_dataset_tag):
        raise ValueError(f"[CKPT] dataset_tag mismatch: ckpt={config.dataset_tag} "
                         f"vs expected={expected_dataset_tag}")
    if expected_scaler_type is not None and config.scaler_type != str(expected_scaler_type):
        raise ValueError(f"[CKPT] scaler_type mismatch: ckpt={config.scaler_type} "
                         f"vs expected={expected_scaler_type}")

    return config, bundle

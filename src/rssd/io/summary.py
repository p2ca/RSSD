"""Run-level metrics summary serialisation."""

from __future__ import annotations

import numpy as np
import os
import re
import torch

__all__ = ["_jsonify", "_float_or_none", "_metrics_summary_dict", "_parse_ckpt_ts"]

def _jsonify(x):
    """Make objects JSON-serializable (Tensor/ndarray/numpy scalar/Path/etc.)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        # keys must be str for stable json
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # fallback: keep reproducibility but avoid crash
    return str(x)


def _float_or_none(x):
    """Convert scalar-like object to float, keep NaN/inf as None for clean csv/json export."""
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _metrics_summary_dict(
    eval_meta: dict,
    logger,
    overall_r2,
    daily_r2_scores,
    reservoir_r2_scores,
    reservoir_r2_daily,
    is_domain_shift: bool,
):
    out = {
        "timestamp": str(logger.timestamp),
        "log_dir": str(logger.log_dir),
        "model_name": str(eval_meta.get("model_name")),
        "exp_name": str(eval_meta.get("exp_name")),
        "model_variant": str(eval_meta.get("model_variant")),
        "source_dataset_tag": str(eval_meta.get("source_dataset_tag")),
        "eval_dataset_tag": str(eval_meta.get("eval_dataset_tag")),
        "source_ckpt_path": str(eval_meta.get("source_ckpt_path")),
        "source_ckpt_ts": str(eval_meta.get("source_ckpt_ts")),
        "scaler_type": str(eval_meta.get("scaler_type")),
        "overall_r2": _float_or_none(overall_r2),
        "daily_r2_scores": [_float_or_none(x) for x in daily_r2_scores],
        "num_target_reservoirs": int(len(reservoir_r2_scores)),
        "target_reservoirs": sorted([str(k) for k in reservoir_r2_scores.keys()]),
        "is_domain_shift": bool(is_domain_shift),
        # the caller supplies these through eval_meta
        "enable_finetune": bool(eval_meta.get("enable_finetune", False)),
        "finetune_mode": str(eval_meta.get("finetune_mode", "disabled")),
        "target_support_samples": eval_meta.get("target_support_samples"),
        "use_darsd": bool(eval_meta.get("use_darsd")),
        "use_res_static": bool(eval_meta.get("use_res_static")),
        "latent_mode": str(eval_meta.get("latent_mode")),
    }
    for i, x in enumerate(daily_r2_scores, start=1):
        out[f"r2_d{i}"] = _float_or_none(x)
    return out


def _parse_ckpt_ts(path: str) -> str:
    """Extract 12-digit timestamp from checkpoint path, if present."""
    if path is None:
        return "unknown"
    s = str(path)
    m = re.search(r"(?:checkpoint_|ckpt_)(\d{12})", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{12})", os.path.basename(s))
    return m.group(1) if m else "unknown"

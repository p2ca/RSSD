"""Small shared helpers: seeding and prediction-tensor shaping."""

from __future__ import annotations

import numpy as np
import random
import torch

__all__ = ["set_seed", "_squeeze_pred_tensor", "link_pred_to_scaled"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _squeeze_pred_tensor(pred: torch.Tensor) -> torch.Tensor:
    """Ensure prediction tensor is (nodes, pred_len) or (B, nodes, pred_len)."""
    if pred is None:
        return pred
    if torch.is_tensor(pred) and pred.dim() == 3 and pred.size(-1) == 1:
        return pred.squeeze(-1)
    return pred


def link_pred_to_scaled(y_hat_raw: torch.Tensor, inv_pack_y: dict) -> torch.Tensor:
    """
    Map model raw output -> scaled-space prediction consistent with training.
    If local y is MinMax, treat model output as logit and apply sigmoid to feature_range.
    """
    if inv_pack_y is None:
        return y_hat_raw
    if inv_pack_y.get("type", None) == "minmax":
        fr_min = float(inv_pack_y["fr_min"])
        fr_max = float(inv_pack_y["fr_max"])
        return fr_min + (fr_max - fr_min) * torch.sigmoid(y_hat_raw)
    return y_hat_raw

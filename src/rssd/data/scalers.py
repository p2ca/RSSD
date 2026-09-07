"""Target scaling: per-reservoir inverse transforms back to physical inflow units."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["_extract_bounds_for_nonneg", "build_local_y_inverse_tensors", "inverse_y_scaled_to_phys_torch", "debug_scaled_impossibility"]

def _extract_bounds_for_nonneg(sc):
    """
    Return a scaled-space lower bound for y>=0, depending on scaler type.

    We only need the condition:
      if y_true >= 0, then y_scaled >= lower_bound(0)

    Supports:
      - StandardScaler: y_scaled = (y - mean) / std
      - MinMaxScaler:  y_scaled = (y - min) / (max - min)  (feature_range assumed [0,1])
      - dict-like with mean/std or min/max
    """
    # sklearn StandardScaler-like
    if hasattr(sc, "mean_") and (hasattr(sc, "scale_") or hasattr(sc, "var_")):
        mean = float(sc.mean_[0]) if np.ndim(sc.mean_) else float(sc.mean_)
        if hasattr(sc, "scale_"):
            std = float(sc.scale_[0]) if np.ndim(sc.scale_) else float(sc.scale_)
        else:
            std = float(np.sqrt(sc.var_[0])) if np.ndim(sc.var_) else float(np.sqrt(sc.var_))
        std = max(std, 1e-6)
        return (0.0 - mean) / std, {"type": "standard", "mean": mean, "std": std}

    # sklearn MinMaxScaler-like
    if hasattr(sc, "data_min_") and hasattr(sc, "data_max_"):
        dmin = float(sc.data_min_[0]) if np.ndim(sc.data_min_) else float(sc.data_min_)
        dmax = float(sc.data_max_[0]) if np.ndim(sc.data_max_) else float(sc.data_max_)
        denom = max(dmax - dmin, 1e-6)

        # feature_range in sklearn default is (0,1), but could be different
        if hasattr(sc, "feature_range"):
            fr_min, fr_max = sc.feature_range
        else:
            fr_min, fr_max = 0.0, 1.0

        # transform formula: X_std=(x-dmin)/(dmax-dmin); X_scaled=X_std*(fr_max-fr_min)+fr_min
        lower = ((0.0 - dmin) / denom) * (fr_max - fr_min) + fr_min
        return lower, {"type": "minmax", "data_min": dmin, "data_max": dmax, "fr": (fr_min, fr_max)}

    # dict-like
    if isinstance(sc, dict):
        if "mean" in sc and ("std" in sc or "scale" in sc):
            mean = float(sc["mean"])
            std = float(sc.get("std", sc.get("scale")))
            std = max(std, 1e-6)
            return (0.0 - mean) / std, {"type": "standard_dict", "mean": mean, "std": std}
        if "min" in sc and "max" in sc:
            dmin = float(sc["min"])
            dmax = float(sc["max"])
            denom = max(dmax - dmin, 1e-6)
            lower = (0.0 - dmin) / denom  # assume feature_range [0,1]
            return lower, {"type": "minmax_dict", "min": dmin, "max": dmax}

    raise TypeError(f"Unknown scaler type: {type(sc)}")


def build_local_y_inverse_tensors(scaler_data, reservoir_names_in_node_order, device):
    """Inverse-transform parameters for every node, as a dict of tensors.

    Reads ``scaler_data['local_scalers_y']``. Both scikit-learn ``MinMaxScaler``
    (the one used by the reported runs) and ``StandardScaler`` are supported.
    """
    local_y = scaler_data["local_scalers_y"]

    # detect type by first scaler
    first_name = reservoir_names_in_node_order[0]
    sc0 = local_y[first_name]

    pack = {}

    # MinMaxScaler-like
    if hasattr(sc0, "data_min_") and hasattr(sc0, "data_max_"):
        dmin, dmax = [], []
        fr_min, fr_max = sc0.feature_range if hasattr(sc0, "feature_range") else (0.0, 1.0)

        for name in reservoir_names_in_node_order:
            sc = local_y[name]
            dmin.append(float(sc.data_min_[0]) if np.ndim(sc.data_min_) else float(sc.data_min_))
            dmax.append(float(sc.data_max_[0]) if np.ndim(sc.data_max_) else float(sc.data_max_))

        pack["type"] = "minmax"
        pack["dmin"] = torch.tensor(dmin, device=device, dtype=torch.float32)     # (nodes,)
        pack["dmax"] = torch.tensor(dmax, device=device, dtype=torch.float32)     # (nodes,)
        pack["fr_min"] = float(fr_min)
        pack["fr_max"] = float(fr_max)
        return pack

    # StandardScaler-like
    if hasattr(sc0, "mean_") and (hasattr(sc0, "scale_") or hasattr(sc0, "var_")):
        mu, std = [], []
        for name in reservoir_names_in_node_order:
            sc = local_y[name]
            m = float(sc.mean_[0]) if np.ndim(sc.mean_) else float(sc.mean_)
            if hasattr(sc, "scale_"):
                s = float(sc.scale_[0]) if np.ndim(sc.scale_) else float(sc.scale_)
            else:
                s = float(np.sqrt(sc.var_[0])) if np.ndim(sc.var_) else float(np.sqrt(sc.var_))
            mu.append(m)
            std.append(max(s, 1e-6))

        pack["type"] = "standard"
        pack["mu"] = torch.tensor(mu, device=device, dtype=torch.float32)        # (nodes,)
        pack["std"] = torch.tensor(std, device=device, dtype=torch.float32)      # (nodes,)
        return pack

    raise TypeError(f"Unsupported local y scaler type: {type(sc0)}")


def inverse_y_scaled_to_phys_torch(y_scaled, inv_pack):
    """Undo the min-max scaling of one batch of targets.

    ``y_scaled`` is ``(nodes, pred_len)`` and ``inv_pack`` comes from
    :func:`build_local_y_inverse_tensors`; the result is inflow in physical units.
    """
    if inv_pack["type"] == "minmax":
        dmin = inv_pack["dmin"][:, None]
        dmax = inv_pack["dmax"][:, None]
        fr_min = inv_pack["fr_min"]
        fr_max = inv_pack["fr_max"]
        denom_fr = max(fr_max - fr_min, 1e-6)
        denom_dm = (dmax - dmin).clamp_min(1e-6)

        # sklearn: X_std=(x-fr_min)/(fr_max-fr_min); X = X_std*(dmax-dmin)+dmin
        y_phys = ((y_scaled - fr_min) / denom_fr) * denom_dm + dmin

    elif inv_pack["type"] == "standard":
        y_phys = y_scaled * inv_pack["std"][:, None] + inv_pack["mu"][:, None]
    else:
        raise RuntimeError("unknown inv_pack type")

    return y_phys


def debug_scaled_impossibility(targets_scaled, scaler_data, reservoir_names_in_node_order, k=15):
    """Flag nodes whose scaled targets fall below what their scaler allows.

    If the raw target is non-negative and ``targets_scaled`` really is ``(y-mean)/std``,
    the minimum of ``targets_scaled`` cannot be below ``(0-mean)/std``. A value under that
    bound means the node was scaled with the wrong scaler, or the targets did not come
    from this scaler at all.
    """
    ts = np.asarray(targets_scaled)  # (S,N,T)
    S, N, T = ts.shape

    local_y = scaler_data["local_scalers_y"]
    mins_scaled = ts.min(axis=(0,2))  # (N,)

    rows = []
    for j in range(N):
        name = reservoir_names_in_node_order[j]
        sc = local_y[name]
        scaled_lower_bound, meta = _extract_bounds_for_nonneg(sc)
        gap = mins_scaled[j] - scaled_lower_bound  # gap < 0 => impossible
        rows.append((gap, j, name, mins_scaled[j], scaled_lower_bound, meta))

    rows.sort(key=lambda x: x[0])  # most negative first
    print("Most impossible nodes (gap=min_scaled - lower_bound):")
    for gap, j, name, msc, lb, meta in rows[:k]:
        print(f"  node={j:03d} {name:>8s} gap={gap: .3f}  min_scaled={msc: .3f}  lb={lb: .3f}  meta={meta}")

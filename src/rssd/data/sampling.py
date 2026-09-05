"""Event-balanced window sampling and flat-window thinning for source training."""

from __future__ import annotations

from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
import numpy as np
import torch

__all__ = ["_resolve_dataset_indices", "_select_low_flow_focus_nodes", "_compute_window_event_scores", "_apply_flat_window_thinning", "_build_event_weighted_sampler"]

def _resolve_dataset_indices(ds):
    if isinstance(ds, Subset):
        parent_idx = _resolve_dataset_indices(ds.dataset)
        return parent_idx[np.asarray(ds.indices, dtype=np.int64)]
    return np.arange(len(ds), dtype=np.int64)


def _select_low_flow_focus_nodes(y_scale_arr, reservoir_names, low_flow_fraction=0.60):
    scales = np.asarray(y_scale_arr, dtype=np.float32)
    finite_mask = np.isfinite(scales) & (scales > 0)
    all_idx = np.arange(len(reservoir_names), dtype=np.int64)
    if not finite_mask.any():
        return all_idx, list(reservoir_names)

    valid_idx = all_idx[finite_mask]
    valid_scales = scales[finite_mask]
    frac = float(np.clip(low_flow_fraction, 0.10, 1.0))
    k = max(1, int(np.ceil(valid_idx.size * frac)))
    order = np.argsort(valid_scales, kind="stable")
    focus_idx = np.sort(valid_idx[order[:k]]).astype(np.int64)
    focus_names = [reservoir_names[int(i)] for i in focus_idx.tolist()]
    return focus_idx, focus_names


def _compute_window_event_scores(y_phys, y_scale_arr, focus_node_idx):
    y = np.asarray(y_phys, dtype=np.float32)
    if y.ndim != 3:
        raise ValueError(f"Expected y_phys shape (N, nodes, pred_len), got {y.shape}")

    focus_node_idx = np.asarray(focus_node_idx, dtype=np.int64)
    if focus_node_idx.size == 0:
        focus_node_idx = np.arange(y.shape[1], dtype=np.int64)

    scale = np.asarray(y_scale_arr, dtype=np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    focus_y = np.abs(y[:, focus_node_idx, :])
    peak = np.nanmax(focus_y, axis=2)
    mean = np.nanmean(focus_y, axis=2)
    rise = np.maximum(y[:, focus_node_idx, -1] - y[:, focus_node_idx, 0], 0.0)

    norm = scale[focus_node_idx][None, :] + 1e-6
    node_scores = 0.60 * (peak / norm) + 0.25 * (mean / norm) + 0.15 * (rise / norm)
    node_scores = np.nan_to_num(node_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return np.max(node_scores, axis=1).astype(np.float32)


def _apply_flat_window_thinning(dataset, window_scores_full, flat_quantile, keep_frac, seed):
    keep_frac = float(np.clip(keep_frac, 0.05, 1.0))
    flat_quantile = float(np.clip(flat_quantile, 0.0, 0.95))

    full_idx = _resolve_dataset_indices(dataset)
    scores = np.asarray(window_scores_full, dtype=np.float32)[full_idx]
    finite_scores = scores[np.isfinite(scores)]

    info = {
        "enabled": bool(keep_frac < 0.999),
        "before": int(len(full_idx)),
        "after": int(len(full_idx)),
        "flat_cutoff": None,
        "flat_count": 0,
        "kept_flat_count": 0,
    }
    if (keep_frac >= 0.999) or (scores.size < 32) or (finite_scores.size < 32):
        return dataset, info

    flat_cutoff = float(np.quantile(finite_scores, flat_quantile))
    flat_mask = np.isfinite(scores) & (scores <= flat_cutoff)
    if not flat_mask.any():
        info["flat_cutoff"] = flat_cutoff
        return dataset, info

    rng = np.random.default_rng(int(seed) + 2025)
    local_keep_idx = []
    kept_flat_count = 0
    for local_idx, is_flat in enumerate(flat_mask.tolist()):
        if (not is_flat) or (rng.random() <= keep_frac):
            local_keep_idx.append(local_idx)
            if is_flat:
                kept_flat_count += 1

    if not local_keep_idx:
        local_keep_idx = [int(np.nanargmax(scores))]
        kept_flat_count = 0

    thinned = Subset(dataset, local_keep_idx)
    info.update({
        "after": int(len(local_keep_idx)),
        "flat_cutoff": flat_cutoff,
        "flat_count": int(flat_mask.sum()),
        "kept_flat_count": int(kept_flat_count),
    })
    return thinned, info


def _build_event_weighted_sampler(dataset, window_scores_full, event_quantile, upweight, seed):
    upweight = max(float(upweight), 1.0)
    event_quantile = float(np.clip(event_quantile, 0.5, 0.99))

    full_idx = _resolve_dataset_indices(dataset)
    scores = np.asarray(window_scores_full, dtype=np.float32)[full_idx]
    finite_scores = scores[np.isfinite(scores)]
    info = {
        "enabled": bool(upweight > 1.0),
        "event_cutoff": None,
        "event_count": 0,
        "num_samples": int(scores.shape[0]),
        "upweight": float(upweight),
    }
    if (upweight <= 1.0) or (scores.size == 0) or (finite_scores.size < 32):
        return None, info

    event_cutoff = float(np.quantile(finite_scores, event_quantile))
    event_mask = np.isfinite(scores) & (scores >= event_cutoff)
    weights = np.ones(scores.shape[0], dtype=np.float64)
    weights[event_mask] = upweight

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(scores.shape[0]),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed) + 3030),
    )
    info.update({
        "event_cutoff": event_cutoff,
        "event_count": int(event_mask.sum()),
    })
    return sampler, info

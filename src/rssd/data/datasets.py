"""Loading the preprocessed reservoir tensors and building the training loaders.

The split, the scaling and the sampling are fixed by the protocol; this module only
loads them and assembles the loaders.

A parsed dataset directory (``data/parsed/<dataset_tag>/``) holds two files:

``all_rsr_data_<scaler_type>.pkl``
    fitted scalers, preprocessing parameters and the per-reservoir chronological
    blocks, from which the validation split is assembled.
``_GNN_supervise_<scaler_type>.pt``
    the windowed training and test tensors plus the graph description
    (``edge_index``, ``encode_map``).
"""

from __future__ import annotations

import os
import pickle
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rssd import paths
from rssd.data.sampling import _apply_flat_window_thinning, _build_event_weighted_sampler
from rssd.data.sampling import _compute_window_event_scores, _select_low_flow_focus_nodes
from rssd.data.scalers import _extract_bounds_for_nonneg
from rssd.utils import WindowDataset
from rssd.utils import build_supervised_split_from_reservoir_blocks
from rssd.utils import collate_zip as _collate
from rssd.utils import inverse_transform_predictions

__all__ = [
    "ParsedDataset", "load_parsed_dataset", "compute_train_scale_stats",
    "build_window_scores", "build_datasets", "build_target_datasets", "build_dataloaders",
    "resolve_embargo_windows",
    "build_alignment_loader",
    "assert_target_scaler_consistency",
]


class ParsedDataset:
    """Everything the training and evaluation loops need from one parsed dataset tag."""

    def __init__(self, dataset_tag, all_rsr_data, scaler_data, X_train, y_train, X_test, y_test,
                 edge_index, encode_map, reservoir_names_in_node_order):
        self.dataset_tag = dataset_tag
        self.all_rsr_data = all_rsr_data
        self.scaler_data = scaler_data
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.edge_index = edge_index
        self.encode_map = encode_map
        self.reservoir_names_in_node_order = reservoir_names_in_node_order

    @property
    def num_nodes(self) -> int:
        return len(self.encode_map)

    @property
    def pred_len(self) -> int:
        return int(self.y_train.shape[-1])

    def state_counts(self):
        """Reservoirs per two-letter state prefix."""
        return Counter(str(n)[:2] for n in self.reservoir_names_in_node_order)

    def auto_run_tag(self) -> str:
        counts = sorted(self.state_counts().items())
        return "keep_" + "_".join(f"{k}{v}" for k, v in counts) + f"_n{self.num_nodes}"


def load_parsed_dataset(dataset_tag: str, scaler_type: str = "local") -> ParsedDataset:
    """Load one parsed dataset tag and validate the contract the model relies on."""
    parsed_path = paths.parsed_dir(dataset_tag)

    with open(os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl"), "rb") as f:
        all_rsr_data = pickle.load(f)
    scaler_data = {
        "scaler_X": all_rsr_data.get("scaler_X"),
        "scaler_y": all_rsr_data.get("scaler_y"),
        "local_scalers_X": all_rsr_data.get("local_scalers_X"),
        "local_scalers_y": all_rsr_data.get("local_scalers_y"),
        "params": all_rsr_data.get("params", {}),
        "diagnostics": all_rsr_data.get("diagnostics", {}),
    }

    supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
    sup_raw = torch.load(supervised_file, map_location="cpu", weights_only=False)
    if not isinstance(sup_raw, dict):
        raise TypeError(f"Unexpected object type in {supervised_file}: {type(sup_raw)}")
    sd = sup_raw["supervised_data"] if "supervised_data" in sup_raw else sup_raw

    graph_data = sup_raw.get("graph_data", None)
    if graph_data is None:
        raise RuntimeError("graph_data missing in _GNN_supervise_*.pt")
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    encode_map = graph_data["encode_map"]
    num_nodes = len(encode_map)

    X_train, y_train = sd["X_train"], sd["y_train"]
    X_test, y_test = sd["X_test"], sd["y_test"]
    if int(X_train.shape[-1]) != 4:
        raise AssertionError(f"Expected 4 input features, got {X_train.shape[-1]}")

    # Reservoir order in NODE INDEX order (critical: every metric is indexed by node).
    names = [None] * num_nodes
    for rsr_name, node_idx in encode_map.items():
        node_idx = int(node_idx)
        if not (0 <= node_idx < num_nodes):
            raise ValueError(f"encode_map contains invalid node_idx={node_idx} for {rsr_name}")
        names[node_idx] = rsr_name
    missing = [i for i, x in enumerate(names) if x is None]
    if missing:
        raise RuntimeError(f"reservoir_names_in_node_order has missing indices: {missing[:20]} ...")

    resolved_scaler_type = scaler_data.get("params", {}).get("scaler_type", scaler_type)
    if resolved_scaler_type == "local":
        local_y = scaler_data.get("local_scalers_y", {})
        miss = [n for n in names if n not in local_y]
        if miss:
            raise RuntimeError(f"local_scalers_y missing {len(miss)} reservoirs, e.g. {miss[:10]}")

    return ParsedDataset(dataset_tag, all_rsr_data, scaler_data, X_train, y_train,
                         X_test, y_test, edge_index, encode_map, names)


def assert_target_scaler_consistency(ds: ParsedDataset, tol: float = 1e-4):
    """Fail fast when the target tensors were not produced by the target's own scalers.

    Physical inflow is non-negative, so every scaled target must sit at or above the value
    its own scaler maps zero to. A node below that bound means the tensors and the scalers
    disagree -- typically a node-order or protocol mix-up -- and every metric computed from
    them would be uninterpretable.
    """
    if ds.scaler_data.get("params", {}).get("scaler_type", "local") != "local":
        return []

    local_y = ds.scaler_data.get("local_scalers_y", {})
    y_test = (ds.y_test.detach().cpu().numpy() if torch.is_tensor(ds.y_test)
              else np.asarray(ds.y_test))
    mins_scaled = np.min(y_test, axis=(0, 2))

    impossible = []
    for j, name in enumerate(ds.reservoir_names_in_node_order):
        lower_bound, meta = _extract_bounds_for_nonneg(local_y[name])
        gap = float(mins_scaled[j] - lower_bound)
        if gap < -tol:
            impossible.append({"node": j, "reservoir": name, "gap": gap,
                               "min_scaled": float(mins_scaled[j]),
                               "lower_bound": float(lower_bound), "scaler": meta})

    if impossible:
        impossible.sort(key=lambda r: r["gap"])
        worst = ", ".join(f"{r['reservoir']}({r['gap']:.4f})" for r in impossible[:5])
        raise RuntimeError(
            f"[SANITY][FATAL] target tensors are inconsistent with the target scalers for "
            f"{len(impossible)}/{ds.num_nodes} reservoirs (worst: {worst}). "
            "This indicates a target-side scaler/order mismatch, so metrics would not be "
            "interpretable.")
    return impossible


def compute_train_scale_stats(ds: ParsedDataset):
    """Per-reservoir physical-space scale (p95-p05) and variance of the training targets.

    The scale feeds the normalised-error diagnostics, the variance feeds the optional
    per-reservoir normalised loss.
    """
    y_train_np = (ds.y_train.detach().cpu().numpy() if torch.is_tensor(ds.y_train)
                  else np.asarray(ds.y_train))
    y_train_orig, _ = inverse_transform_predictions(
        predictions=y_train_np, targets=y_train_np,
        scaler_data=ds.scaler_data, encode_map=ds.encode_map,
    )

    num_nodes = ds.num_nodes
    scale_arr = np.zeros((num_nodes,), dtype=np.float32)
    var_arr = np.zeros((num_nodes,), dtype=np.float32)
    for j in range(num_nodes):
        vals = y_train_orig[:, j, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            scale_arr[j] = np.nan
            var_arr[j] = 1e-6
            continue
        scale_arr[j] = max(float(np.quantile(vals, 0.95) - np.quantile(vals, 0.05)), 1e-6)
        var_arr[j] = max(float(np.var(vals)), 1e-6)

    scale_dict = {ds.reservoir_names_in_node_order[j]: float(scale_arr[j]) for j in range(num_nodes)}
    return scale_arr, scale_dict, var_arr, y_train_orig


def build_window_scores(ds: ParsedDataset, y_train_orig, train_y_scale_arr,
                        low_flow_fraction: float = 0.60):
    """Event scores per training window, plus the low-flow focus nodes they are scored on."""
    focus_idx, focus_names = _select_low_flow_focus_nodes(
        train_y_scale_arr, ds.reservoir_names_in_node_order, low_flow_fraction=low_flow_fraction)
    scores = _compute_window_event_scores(y_train_orig, train_y_scale_arr, focus_idx)
    return scores, focus_idx, focus_names


def resolve_embargo_windows(ds: ParsedDataset) -> int:
    """Windows dropped at the head of the validation and test blocks.

    ``Tin + horizon - 1`` is the number of leading windows whose input history would
    otherwise reach back into the preceding block.
    """
    return int(ds.X_train.shape[1]) + int(ds.pred_len) - 1


def build_datasets(ds: ParsedDataset):
    """Train / validation / test datasets under the frozen split protocol.

    The parsed record is divided chronologically into training, validation and test
    blocks. Every training window is used to fit the model; the validation block
    selects the checkpoint and the test block scores it. Both of the latter are
    embargoed at the head so that no window they contain reads days that belong to
    the preceding block.
    """
    purge = resolve_embargo_windows(ds)

    train_dataset_full = WindowDataset(ds.X_train, ds.y_train, ds.edge_index)
    test_dataset = WindowDataset(ds.X_test[purge:], ds.y_test[purge:], ds.edge_index)
    X_val, y_val = build_supervised_split_from_reservoir_blocks(
        ds.all_rsr_data, ds.reservoir_names_in_node_order, "val",
        purge_head_windows=purge,
    )
    train_dataset = Subset(train_dataset_full, list(range(len(train_dataset_full))))
    val_dataset = WindowDataset(X_val, y_val, ds.edge_index)
    print(f"[SPLIT] train={len(train_dataset)} val={len(val_dataset)} "
          f"test={len(test_dataset)} embargo={purge}")

    return train_dataset, val_dataset, test_dataset


def build_target_datasets(ds: ParsedDataset):
    """Support / validation / test datasets for a target reservoir set.

    The adaptation ("support") set is the target's complete training block. Adaptation
    is stopped on the target's own validation block and the forecast scores are
    computed on its test block; both are embargoed at the head.

    Returns ``(support, val, test)``.
    """
    purge = resolve_embargo_windows(ds)

    support = WindowDataset(ds.X_train, ds.y_train, ds.edge_index)
    X_val, y_val = build_supervised_split_from_reservoir_blocks(
        ds.all_rsr_data, ds.reservoir_names_in_node_order, "val",
        purge_head_windows=purge,
    )
    val = WindowDataset(X_val, y_val, ds.edge_index)
    test = WindowDataset(ds.X_test[purge:], ds.y_test[purge:], ds.edge_index)
    return support, val, test


def _thin_and_drop(train_dataset, window_scores, *, use_event_balanced_sampling: bool,
                   flat_window_quantile: float, flat_window_keep_frac: float,
                   train_window_drop_frac: float, train_window_drop_seed: int, seed: int):
    info = {}
    if use_event_balanced_sampling:
        train_dataset, flat_thin_info = _apply_flat_window_thinning(
            train_dataset, window_scores, flat_quantile=flat_window_quantile,
            keep_frac=flat_window_keep_frac, seed=seed)
        info["window_thinning"] = flat_thin_info

    drop_frac = max(0.0, min(float(train_window_drop_frac), 0.95))
    if drop_frac > 0.0:
        n = len(train_dataset)
        n_keep = max(1, int(round(n * (1.0 - drop_frac))))
        g_mask = torch.Generator().manual_seed(int(train_window_drop_seed))
        keep_idx = torch.randperm(n, generator=g_mask).tolist()[:n_keep]
        train_dataset = Subset(train_dataset, keep_idx)
        info["lowdata"] = {"drop_frac": drop_frac, "keep": n_keep, "of": n}

    return train_dataset, info


def build_dataloaders(train_dataset, val_dataset, test_dataset, window_scores, *,
                      batch_size: int = 128, seed: int = 42,
                      use_event_balanced_sampling: bool = False,
                      event_score_quantile: float = 0.80, event_upweight: float = 1.0,
                      flat_window_quantile: float = 0.0, flat_window_keep_frac: float = 1.0,
                      train_window_drop_frac: float = 0.0, train_window_drop_seed: int = 0,
                      num_workers: int = 0, pin_memory=None, persistent_workers=False,
                      prefetch_factor=None, multiprocessing_context=None):
    """Loaders for training, in-training diagnostics, validation and test."""
    train_dataset, info = _thin_and_drop(
        train_dataset, window_scores,
        use_event_balanced_sampling=use_event_balanced_sampling,
        flat_window_quantile=flat_window_quantile,
        flat_window_keep_frac=flat_window_keep_frac,
        train_window_drop_frac=train_window_drop_frac,
        train_window_drop_seed=train_window_drop_seed, seed=seed)

    train_sampler = None
    if use_event_balanced_sampling:
        train_sampler, sampler_info = _build_event_weighted_sampler(
            train_dataset, window_scores, event_quantile=event_score_quantile,
            upweight=event_upweight, seed=seed)
        info["window_sampler"] = sampler_info

    loader_kwargs = dict(
        batch_size=batch_size, collate_fn=_collate, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor, multiprocessing_context=multiprocessing_context,
    )
    if train_sampler is None:
        train_loader = DataLoader(train_dataset, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed + 123),
                                  **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=False, sampler=train_sampler,
                                  **loader_kwargs)

    # Diagnostics run on the post-split training set so nothing leaks from validation.
    train_loader_diag = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return {
        "train": train_loader, "train_diag": train_loader_diag,
        "val": val_loader, "test": test_loader,
        "train_dataset": train_dataset, "info": info,
    }


def build_alignment_loader(target_dataset_tag: str, *, scaler_type: str = "local",
                           batch_size: int = 128, seed: int = 42, num_workers: int = 0,
                           pin_memory=None, persistent_workers=False, prefetch_factor=None,
                           multiprocessing_context=None):
    """Unlabelled target-support loader used by the MMD / CORAL / DANN baselines."""
    target_parsed_path = paths.parsed_dir(target_dataset_tag)
    target_supervised_file = os.path.join(target_parsed_path, f"_GNN_supervise_{scaler_type}.pt")
    tgt_sup_raw = torch.load(target_supervised_file, map_location="cpu", weights_only=False)
    if not isinstance(tgt_sup_raw, dict):
        raise TypeError(f"Unexpected target supervised object type in "
                        f"{target_supervised_file}: {type(tgt_sup_raw)}")

    tgt_sd = tgt_sup_raw["supervised_data"] if "supervised_data" in tgt_sup_raw else tgt_sup_raw
    tgt_graph_data = tgt_sup_raw.get("graph_data", None)
    if tgt_graph_data is None:
        raise RuntimeError(f"graph_data missing in target supervised file: {target_supervised_file}")
    target_edge_index = torch.tensor(tgt_graph_data["edge_index"], dtype=torch.long)

    # labels are only needed to satisfy the dataset wrapper; the alignment losses ignore them
    target_dataset = WindowDataset(tgt_sd["X_train"], tgt_sd["y_train"], target_edge_index)
    return DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor, multiprocessing_context=multiprocessing_context,
        generator=torch.Generator().manual_seed(seed + 456),
    ), target_dataset

"""Building a parsed dataset from the aligned per-reservoir daily records.

This is the step in front of everything else: it turns ``align/<id>.csv`` into the
windowed tensors and fitted scalers that :mod:`rssd.data.datasets` loads.

Contract of one aligned record (``align/<id>.csv``), one row per day:

===========  ==========================================================
``date``     calendar date, parseable by :func:`pandas.to_datetime`
``inflow``   reservoir inflow, the forecast target
``precip``   precipitation
``tmax``     maximum air temperature
``tmin``     minimum air temperature
``storage``  reservoir storage; read only when the static attributes are built
``elevation``  water-surface elevation; likewise
===========  ==========================================================

What the step does, in order:

1. keep the reservoirs named in ``reservoirs_<dataset tag>.txt``;
2. for a target pool, keep only that reservoir's most recent
   :data:`TARGET_HISTORY_YEARS` of record;
3. repair non-positive and non-finite inflow, which marks missing record rather than a
   real value, and clip negative precipitation to zero;
4. cut sliding windows of ``days_x`` input days and ``days_y`` forecast days, one window
   per day;
5. split each reservoir's windows chronologically into 70 / 15 / 15;
6. fit one min-max scaler per reservoir on its own training window, and apply it to all
   three blocks;
7. write ``all_rsr_data_local.pkl`` (blocks, scalers, parameters) and
   ``_GNN_supervise_local.pt`` (the stacked training and test tensors).
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

__all__ = [
    "FEATURE_COLUMNS", "TARGET_HISTORY_YEARS",
    "fill_nonpositive_runs", "truncate_recent_history_by_years",
    "create_sliding_windows", "split_train_val_test", "prepare_input_supervised",
    "preprocess_reservoir_data", "build_parsed_dataset",
]

# The four dynamic inputs, in the order the tensors are indexed by. inflow is first
# because it is both an input and the forecast target.
FEATURE_COLUMNS = ["inflow", "precip", "tmax", "tmin"]
INFLOW_COLUMN_INDEX = FEATURE_COLUMNS.index("inflow")

# Attribute columns read from the same record when the static matrix is assembled.
RECORD_COLUMNS = FEATURE_COLUMNS + ["storage", "elevation"]

# Frozen protocol: a target reservoir contributes only its most recent decade.
TARGET_HISTORY_YEARS = 10


def fill_nonpositive_runs(x: np.ndarray, *, invalid_leq: float = 0.0) -> np.ndarray:
    """Replace each run of non-positive or non-finite values by its neighbours' mean.

    Zero and negative inflow marks a gap in the record rather than a measurement, so a
    contiguous invalid run is filled with the average of the nearest valid value on each
    side; a run touching either end falls back to the one neighbour it has. Only inflow
    is treated this way — zero precipitation and negative temperature are meaningful.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    invalid = (~np.isfinite(x)) | (x <= invalid_leq)
    if not invalid.any():
        return x.astype(np.float32)

    idx = np.where(invalid)[0]
    starts = idx[np.r_[True, np.diff(idx) > 1]]
    ends = idx[np.r_[np.diff(idx) > 1, True]]

    n = len(x)
    for s_i, e_i in zip(starts, ends):
        left = x[s_i - 1] if s_i - 1 >= 0 else np.nan
        right = x[e_i + 1] if e_i + 1 < n else np.nan

        if np.isfinite(left) and np.isfinite(right):
            fill = 0.5 * (left + right)
        elif np.isfinite(left):
            fill = left
        elif np.isfinite(right):
            fill = right
        else:
            fill = 0.0
        x[s_i:e_i + 1] = fill

    return x.astype(np.float32)


def truncate_recent_history_by_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Keep the most recent ``years`` of one reservoir's record, relative to its own end."""
    if years is None:
        return df.reset_index(drop=True)

    years = int(years)
    if years <= 0:
        raise ValueError(f"years must be positive, got {years}")
    if df.empty:
        return df.reset_index(drop=True)

    end_date = df["date"].max()
    start_date = end_date - pd.DateOffset(years=years)
    return df[df["date"] >= start_date].copy().reset_index(drop=True)


def create_sliding_windows(data: np.ndarray, days_x: int, days_y: int,
                           inflow_col_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Windows of ``days_x`` input days followed by ``days_y`` forecast days.

    ``data`` is ``(T, F)``; the result is ``X`` of ``(N, days_x, F)`` and ``y`` of
    ``(N, days_y)`` holding the inflow column only. One window starts on every day of the
    record for which a complete input block and forecast block exist.
    """
    T, F = data.shape
    if T < days_x + days_y:
        return (np.empty((0, days_x, F), dtype=np.float32),
                np.empty((0, days_y), dtype=np.float32))

    n_windows = T - days_x - days_y + 1
    X = np.stack([data[s:s + days_x, :] for s in range(n_windows)]).astype(np.float32)
    y = np.stack([data[s + days_x:s + days_x + days_y, inflow_col_idx]
                  for s in range(n_windows)]).astype(np.float32)
    return X, y


def _compute_split_points(N: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    """``(n_train, n_val_end)``: train is ``[0, n_train)``, val ``[n_train, n_val_end)``."""
    n_train = int(N * train_ratio)
    n_val_end = int(N * (train_ratio + val_ratio))
    if N >= 3:                                   # keep every block non-empty where possible
        n_train = max(1, min(n_train, N - 2))
        n_val_end = max(n_train + 1, min(n_val_end, N - 1))
    return n_train, n_val_end


def split_train_val_test(X: np.ndarray, y: np.ndarray, train_ratio: float,
                         val_ratio: float) -> Dict[str, Dict[str, np.ndarray]]:
    """Chronological split of one reservoir's windows: no shuffling, no leakage."""
    N = X.shape[0]
    if N < 3:
        empty_x = np.empty((0,), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        return {split: {"X": empty_x, "y": empty_y} for split in ("train", "val", "test")}

    n_train, n_val_end = _compute_split_points(N, train_ratio, val_ratio)
    return {
        "train": {"X": X[:n_train], "y": y[:n_train]},
        "val": {"X": X[n_train:n_val_end], "y": y[n_train:n_val_end]},
        "test": {"X": X[n_val_end:], "y": y[n_val_end:]},
    }


def prepare_input_supervised(input_data: Dict, encode_map: Dict[str, int], split: str):
    """Stack the per-reservoir blocks into ``X (N, days_x, nodes, F)`` and ``y (N, nodes, days_y)``.

    Windows containing a non-finite value are dropped, and every reservoir is truncated to
    the shortest surviving length so the node axis stays rectangular.
    """
    xs, ys = [], []
    for name in list(encode_map.keys()):
        node_dict = input_data[name][split]
        x = torch.tensor(node_dict["X"], dtype=torch.float32)
        y = torch.tensor(node_dict["y"], dtype=torch.float32)

        mask = ~(torch.isnan(x).any(dim=1).any(dim=1) | torch.isnan(y).any(dim=1))
        x, y = x[mask], y[mask]

        if x.shape[0] == 0:
            print(f"[WARN] {name} split={split} has 0 samples after NaN filtering, skip.")
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        raise RuntimeError(f"No valid samples left for split={split} after NaN filtering.")

    min_len = min(t.shape[0] for t in xs)
    X = torch.stack([t[:min_len] for t in xs], dim=1)      # (N, nodes, days_x, F)
    y = torch.stack([t[:min_len] for t in ys], dim=1)      # (N, nodes, days_y)
    return X.transpose(1, 2).contiguous(), y               # (N, days_x, nodes, F)


def _oob_fraction(arr: np.ndarray, low: float = 0.0, high: float = 1.0) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean((arr < low) | (arr > high)))


def _read_reservoir_list(list_file) -> List[str]:
    """Reservoir identifiers, one per line; blank lines and ``#`` comments ignored."""
    names = []
    with open(list_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    if not names:
        raise ValueError(f"reservoir list is empty: {list_file}")
    return names


def preprocess_reservoir_data(
    align_dir: str,
    reservoir_names: List[str],
    *,
    days_x: int = 30,
    days_y: int = 7,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    role: str = "source",
    target_history_years: int = TARGET_HISTORY_YEARS,
) -> Tuple[Dict, List[str]]:
    """Window, split and scale every named reservoir. Returns ``(blocks, node order)``."""
    role = (role or "source").strip().lower()
    if role not in ("source", "target"):
        raise ValueError(f"role must be 'source' or 'target', got {role!r}")
    if not os.path.isdir(align_dir):
        raise FileNotFoundError(f"aligned-record directory not found: {align_dir}")

    raw_per_reservoir: Dict[str, Dict] = {}

    for name in tqdm(reservoir_names, desc="Building sliding windows"):
        csv_path = os.path.join(align_dir, f"{name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"aligned record not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "date" not in df.columns:
            raise KeyError(f"{csv_path} missing 'date' column.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if role == "target":
            before = len(df)
            df = truncate_recent_history_by_years(df, years=target_history_years)
            print(f"[INFO] {name}: kept the most recent {target_history_years} years, "
                  f"{before} -> {len(df)} rows")

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"{csv_path} missing required column(s): {missing}")

        values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        inflow = values[:, INFLOW_COLUMN_INDEX]
        bad = int(((~np.isfinite(inflow)) | (inflow < 0.0)).sum())
        if bad > 0:
            values[:, INFLOW_COLUMN_INDEX] = fill_nonpositive_runs(inflow, invalid_leq=-1e-12)
            print(f"[WARN] {name}: {bad} non-positive or non-finite inflow values repaired")

        precip_idx = FEATURE_COLUMNS.index("precip")
        negative_precip = int((values[:, precip_idx] < 0).sum())
        if negative_precip > 0:
            print(f"[WARN] {name}: {negative_precip} negative precipitation values clipped to 0")
            values[values[:, precip_idx] < 0, precip_idx] = 0.0

        X, y = create_sliding_windows(data=values, days_x=days_x, days_y=days_y,
                                      inflow_col_idx=INFLOW_COLUMN_INDEX)

        if X.shape[0] == 0:
            print(f"[WARN] {name}: no valid windows, skipped.")
            continue

        raw_per_reservoir[name] = split_train_val_test(X, y, train_ratio, val_ratio)

    if not raw_per_reservoir:
        raise RuntimeError("no reservoir produced a valid window")

    # One min-max scaler per reservoir, fitted on that reservoir's own training block.
    processed: Dict = {}
    local_scalers_X: Dict[str, MinMaxScaler] = {}
    local_scalers_y: Dict[str, MinMaxScaler] = {}
    scaler_oob_report: Dict[str, Dict] = {}

    for name, blocks in raw_per_reservoir.items():
        X_train, y_train = blocks["train"]["X"], blocks["train"]["y"]
        if X_train.size == 0 or y_train.size == 0:
            print(f"[WARN] {name}: empty training block, cannot fit a scaler, skipped.")
            continue

        scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X_train.reshape(-1, X_train.shape[-1]))
        scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(y_train.reshape(-1, 1))
        local_scalers_X[name] = scaler_X
        local_scalers_y[name] = scaler_y

        scaled_blocks, y_scaled_blocks = {}, {}
        for split in ("train", "val", "test"):
            X_split, y_split = blocks[split]["X"], blocks[split]["y"]
            if X_split.size == 0:
                scaled_blocks[split] = {
                    "X": np.empty((0, days_x, len(FEATURE_COLUMNS)), dtype=np.float32),
                    "y": np.empty((0, days_y), dtype=np.float32)}
                y_scaled_blocks[split] = np.empty((0, days_y), dtype=np.float32)
                continue

            N, Tx, F = X_split.shape
            X_scaled = scaler_X.transform(X_split.reshape(-1, F)).reshape(N, Tx, F)
            y_scaled = scaler_y.transform(y_split.reshape(-1, 1)).reshape(y_split.shape)

            scaled_blocks[split] = {"X": X_scaled.astype(np.float32),
                                    "y": y_scaled.astype(np.float32)}
            y_scaled_blocks[split] = y_scaled.astype(np.float32)

        scaler_oob_report[name] = {
            f"y_{split}_oob": _oob_fraction(y_scaled_blocks[split])
            for split in ("train", "val", "test")
        }
        processed[name] = scaled_blocks

    node_order = sorted(processed.keys())
    processed["scaler_X"] = None
    processed["scaler_y"] = None
    processed["local_scalers_X"] = local_scalers_X
    processed["local_scalers_y"] = local_scalers_y
    processed["params"] = {
        "input_features": len(FEATURE_COLUMNS),
        "days_x": days_x,
        "days_y": days_y,
        "scaler_type": "local",
        "feature_cols": list(FEATURE_COLUMNS),
        "inflow_col_idx": INFLOW_COLUMN_INDEX,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "role": role,
        "target_history_years": int(target_history_years),
    }
    processed["diagnostics"] = {"scaler_oob_report": scaler_oob_report}
    return processed, node_order


def build_parsed_dataset(
    dataset_tag: str,
    *,
    data_root=None,
    reservoir_list_file=None,
    output_dir=None,
    role: str = "source",
    **kwargs,
) -> str:
    """Write ``parsed/<dataset tag>/`` from ``<data root>/align`` and the pool list.

    ``data_root`` defaults to the directory :mod:`rssd.paths` resolves, and
    ``reservoir_list_file`` to ``<data root>/reservoirs_<dataset tag>.txt``.
    """
    from rssd import paths

    data_root = os.path.abspath(str(data_root)) if data_root else str(paths.DATA_DIR)
    align_dir = os.path.join(data_root, "align")
    list_file = str(reservoir_list_file or os.path.join(data_root,
                                                        f"reservoirs_{dataset_tag}.txt"))
    output_dir = str(output_dir or os.path.join(data_root, "parsed", dataset_tag))
    os.makedirs(output_dir, exist_ok=True)

    reservoir_names = _read_reservoir_list(list_file)
    print(f"[INFO] {dataset_tag}: {len(reservoir_names)} reservoirs from {list_file}")

    processed, node_order = preprocess_reservoir_data(
        align_dir, reservoir_names, role=role, **kwargs)

    encode_map = {name: idx for idx, name in enumerate(node_order)}
    num_nodes = len(node_order)
    graph_data = {
        "A": np.eye(num_nodes, dtype=np.float32),
        "edge_index": np.vstack([np.arange(num_nodes), np.arange(num_nodes)]),
        "encode_map": encode_map,
    }

    X_train, y_train = prepare_input_supervised(processed, encode_map, "train")
    X_test, y_test = prepare_input_supervised(processed, encode_map, "test")

    with open(os.path.join(output_dir, "all_rsr_data_local.pkl"), "wb") as f:
        pickle.dump(processed, f)
    torch.save({"graph_data": graph_data,
                "supervised_data": {"X_train": X_train, "y_train": y_train,
                                    "X_test": X_test, "y_test": y_test}},
               os.path.join(output_dir, "_GNN_supervise_local.pt"))

    print(f"[INFO] {dataset_tag}: {num_nodes} nodes | train={X_train.shape[0]} "
          f"test={X_test.shape[0]} -> {output_dir}")
    return output_dir

"""Structured run outputs: the per-reservoir metric CSV."""

from __future__ import annotations

import csv
import math
import os

__all__ = ["save_reservoir_metrics_csv"]

def save_reservoir_metrics_csv(
    out_csv_path,
    reservoir_names,
    reservoir_r2_dict,
    reservoir_r2_daily_dict=None,
    drift_z_arr=None,
    drift_l2_arr=None,
    drift_md_mean_arr=None,
    mean_norm_train_arr=None,
    mean_norm_test_arr=None,
    ood_mean_arr=None,
    ood_q90_arr=None,
    mae7_scaled_mean_arr=None,
):
    """
    Write per-reservoir metrics to CSV for later latent-space analysis.

    reservoir_names: list[str] in node order
    reservoir_r2_dict: dict[str, float] mapping reservoir_name -> R2 (flatten over horizon)
    reservoir_r2_daily_dict: dict[str, list[float]] mapping reservoir_name -> [r2_d1..r2_dN]
    """
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    def _fmt(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if math.isnan(v) or math.isinf(v):
            return ""
        return v

    def _get(metric, name, i):
        if metric is None:
            return ""
        if isinstance(metric, dict):
            return _fmt(metric.get(name, float("nan")))
        try:
            return _fmt(metric[i])
        except Exception:
            return ""

    # infer horizon length for daily R2 columns
    n_days = None
    if isinstance(reservoir_r2_daily_dict, dict) and len(reservoir_r2_daily_dict) > 0:
        first = next(iter(reservoir_r2_daily_dict.values()))
        try:
            n_days = int(len(first))
        except Exception:
            n_days = None

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        header = [
            "reservoir",
            "state_prefix",
            "r2",
            "drift_z",
            "drift_l2",
            "drift_md_mean",
            "mean_norm_train",
            "mean_norm_test",
            "ood_mean",
            "ood_q90",
            "mae7_scaled_mean",
        ]
        if n_days is not None:
            header += [f"r2_d{d+1}" for d in range(n_days)]
        w.writerow(header)

        for i, name in enumerate(reservoir_names):
            r2 = reservoir_r2_dict.get(name, float("nan"))

            row = [
                name,
                name[:2],
                _fmt(r2),
                _get(drift_z_arr, name, i),
                _get(drift_l2_arr, name, i),
                _get(drift_md_mean_arr, name, i),
                _get(mean_norm_train_arr, name, i),
                _get(mean_norm_test_arr, name, i),
                _get(ood_mean_arr, name, i),
                _get(ood_q90_arr, name, i),
                _get(mae7_scaled_mean_arr, name, i),
            ]

            if n_days is not None:
                r2_list = reservoir_r2_daily_dict.get(name, [float("nan")] * n_days) if reservoir_r2_daily_dict else [float("nan")] * n_days
                # pad/trim defensively
                r2_list = list(r2_list)[:n_days] + [float("nan")] * max(0, n_days - len(r2_list))
                row += [_fmt(v) for v in r2_list]

            w.writerow(row)

"""The static reservoir attributes, assembled in node-index order.

Supported attributes, in the default order used by the v2 protocol:

===  ====================  ==========================================================
idx  attribute             source
===  ====================  ==========================================================
0    ``storage_max``       maximum storage in the reservoir's aligned daily record
1    ``elev_mean``         mean water-surface elevation in the same record
2    ``surface_area``      surface area (km^2) from the reservoir attribute table
3    ``lat``               latitude
4    ``lon``               longitude
5    ``ground_elev``       catchment ground elevation
===  ====================  ==========================================================

A checkpoint stores the attribute list it was trained with, so evaluation always rebuilds
the matrix from ``config["meta_feature_names"]`` rather than from the default order.

Normalisation is robust (median / IQR). Which reservoirs define those statistics matters:

* source training normalises with the statistics of its own reservoirs;
* evaluation normalises the target reservoirs with the **source** statistics, so that a
  checkpoint sees attributes on the scale it was trained on. Under cross-regime transfer
  this can push target attributes far outside the source range, which is why the caller
  may clip the normalised matrix (``CROSS_STATIC_NORM_CLIP``).

The assembly and
normalisation below are the functional form of the same cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from rssd import paths

__all__ = [
    "STATIC_ATTRIBUTE_NAMES", "LEGACY_STATIC_ATTRIBUTE_NAMES", "STATIC_EPS",
    "_safe_nanmax", "_safe_nanmean", "load_latlon_map",
    "build_static_matrix_raw", "robust_norm_stats", "apply_norm", "build_static_matrix",
]

STATIC_ATTRIBUTE_NAMES = ("storage_max", "elev_mean", "surface_area", "lat", "lon", "ground_elev")
# what checkpoints predating the surface-area addition (2026-05-26) carry
LEGACY_STATIC_ATTRIBUTE_NAMES = ("storage_max", "elev_mean", "lat", "lon", "ground_elev")
STATIC_EPS = 1e-6

LATLON_FILENAME = "reservoir_latlon_elev_surface_area.csv"
_RECORD_DERIVED = {"storage_max", "elev_mean"}
_TABLE_DERIVED = {"lat", "lon", "ground_elev", "surface_area"}


def _safe_nanmax(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size > 0 else 0.0


def _safe_nanmean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def load_latlon_map(latlon_csv=None):
    """``{NIDID: {lat, lon, ground_elev, surface_area}}`` from the reservoir attribute table."""
    latlon_csv = latlon_csv or (paths.meta_dir() / LATLON_FILENAME)
    latlon_df = pd.read_csv(latlon_csv)
    for column in ("ELEV_M", "SURFACE_AREA_KM2"):
        if column not in latlon_df.columns:
            raise KeyError(f"[res_static] {latlon_csv} must contain {column}. "
                           f"Got cols: {list(latlon_df.columns)}")
    return {
        str(r["NIDID"]).strip(): {
            "lat": float(r["LATITUDE"]),
            "lon": float(r["LONGITUDE"]),
            "ground_elev": float(r["ELEV_M"]),
            "surface_area": float(r["SURFACE_AREA_KM2"]) if pd.notna(r["SURFACE_AREA_KM2"]) else 0.0,
        }
        for _, r in latlon_df.iterrows()
    }


def build_static_matrix_raw(reservoir_names, feature_names=STATIC_ATTRIBUTE_NAMES,
                            align_dir=None, latlon_csv=None, latlon_map=None):
    """Un-normalised ``(len(reservoir_names), len(feature_names))`` matrix, in node order."""
    feature_names = list(feature_names)
    unknown = [f for f in feature_names if f not in _RECORD_DERIVED | _TABLE_DERIVED]
    if unknown:
        raise ValueError(f"[res_static] unsupported attribute(s): {unknown}")

    align_dir = str(align_dir or paths.align_dir())
    latlon_map = latlon_map if latlon_map is not None else load_latlon_map(latlon_csv)
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    needs_record = bool(_RECORD_DERIVED & set(feature_names))

    X = np.zeros((len(reservoir_names), len(feature_names)), dtype=np.float32)
    for i, rid in enumerate(reservoir_names):
        if needs_record:
            fp = os.path.join(align_dir, f"{rid}.csv")
            if not os.path.exists(fp):
                raise FileNotFoundError(f"[res_static] align file not found: {fp}")
            df = pd.read_csv(fp)
            if "storage" not in df.columns or "elevation" not in df.columns:
                raise KeyError(f"[res_static] {fp} must contain storage/elevation. "
                               f"Got cols: {list(df.columns)[:20]}")
            if "storage_max" in feat_idx:
                X[i, feat_idx["storage_max"]] = _safe_nanmax(df["storage"].to_numpy())
            if "elev_mean" in feat_idx:
                X[i, feat_idx["elev_mean"]] = _safe_nanmean(df["elevation"].to_numpy())

        if rid not in latlon_map:
            raise KeyError(f"[res_static] lat/lon missing for {rid}")
        entry = latlon_map[rid]
        for name in _TABLE_DERIVED:
            if name in feat_idx:
                X[i, feat_idx[name]] = float(entry[name])

    return X


def robust_norm_stats(raw):
    """Median and inter-quartile range per attribute, computed across reservoirs."""
    raw = np.asarray(raw, dtype=np.float32)
    med = np.median(raw, axis=0)
    iqr = np.percentile(raw, 75, axis=0) - np.percentile(raw, 25, axis=0)
    return med, iqr


def apply_norm(raw, med, iqr, eps: float = STATIC_EPS, clip=None):
    """Apply median/IQR normalisation, optionally clipping to ``+/- clip``."""
    out = (np.asarray(raw, dtype=np.float32) - med) / (iqr + eps)
    if clip is not None and float(clip) > 0.0:
        out = np.clip(out, -float(clip), float(clip))
    return out.astype(np.float32)


def build_static_matrix(reservoir_names, feature_names=STATIC_ATTRIBUTE_NAMES,
                        reference_names=None, align_dir=None, latlon_csv=None,
                        eps: float = STATIC_EPS, clip=None):
    """Normalised attribute matrix plus the pieces needed to audit it.

    Parameters
    ----------
    reservoir_names
        Reservoirs to build rows for, in node-index order.
    reference_names
        Reservoirs whose statistics define the normalisation. ``None`` (the training case)
        uses ``reservoir_names`` itself; evaluation passes the source reservoirs.

    Returns
    -------
    dict with ``normalized``, ``raw``, ``med``, ``iqr`` and the ``abs_max`` / ``abs_p95`` /
    ``abs_p99`` drift diagnostics of the normalised matrix before clipping.
    """
    latlon_map = load_latlon_map(latlon_csv)
    raw = build_static_matrix_raw(reservoir_names, feature_names, align_dir, latlon_map=latlon_map)

    if reference_names is None:
        ref_raw = raw
    else:
        ref_raw = build_static_matrix_raw(reference_names, feature_names, align_dir,
                                          latlon_map=latlon_map)
    med, iqr = robust_norm_stats(ref_raw)

    unclipped = apply_norm(raw, med, iqr, eps=eps, clip=None)
    return {
        "normalized": apply_norm(raw, med, iqr, eps=eps, clip=clip),
        "raw": raw,
        "med": med,
        "iqr": iqr,
        "abs_max": float(np.max(np.abs(unclipped))),
        "abs_p95": float(np.percentile(np.abs(unclipped), 95)),
        "abs_p99": float(np.percentile(np.abs(unclipped), 99)),
    }

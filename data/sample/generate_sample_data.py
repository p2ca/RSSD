"""Regenerate the synthetic sample bundle shipped in this directory.

    python data/sample/generate_sample_data.py

The records written here are **simulated**, not observations. They exist so that the
preprocessing, training and evaluation commands can be run end to end without the
reservoir database, and they follow exactly the column contract the pipeline expects:

``align/<id>.csv``
    ``date, inflow, storage, elevation, precip, tmax, tmin`` — one row per day.
``meta/reservoir_latlon_elev_surface_area.csv``
    ``NIDID, LATITUDE, LONGITUDE, ELEV_M, SURFACE_AREA_KM2`` — one row per reservoir.

The generator is deterministic: the same seed reproduces the shipped files byte for byte.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
START = "2010-01-01"
END = "2021-12-31"
SEED = 20260907

# Five simulated reservoirs. The first three stand in for a source pool and the last two
# for a target pool; "snowmelt" and "rainfall" describe the shape of the simulated
# hydrograph, not a real hydroclimatic classification.
RESERVOIRS = [
    # id,        regime,      mean inflow, storm response, latitude, longitude, elevation, area
    ("SYN0001", "snowmelt", 320.0, 0.35, 39.90, -105.60, 2480.0, 12.4),
    ("SYN0002", "snowmelt", 180.0, 0.30, 41.20, -106.10, 2210.0, 7.8),
    ("SYN0003", "rainfall", 640.0, 0.85, 34.40, -092.70, 120.0, 31.5),
    ("SYN0004", "snowmelt", 240.0, 0.32, 40.60, -105.10, 2620.0, 9.1),
    ("SYN0005", "rainfall", 480.0, 0.80, 33.10, -093.40, 95.0, 22.7),
]

SOURCE_POOL = ["SYN0001", "SYN0002", "SYN0003"]
TARGET_POOL = ["SYN0004", "SYN0005"]


def _seasonal(day_of_year, peak_day, amplitude):
    return amplitude * np.cos(2.0 * np.pi * (day_of_year - peak_day) / 365.25)


def _simulate(rng, dates, regime, mean_inflow, storm_response, ground_elev):
    """One reservoir's daily record."""
    n = len(dates)
    doy = dates.dayofyear.to_numpy(dtype=np.float64)

    # --- air temperature: seasonal cycle plus weather noise -------------------
    temp_mean = 12.0 - 0.004 * ground_elev + _seasonal(doy, peak_day=196, amplitude=13.0)
    weather = np.zeros(n)
    for t in range(1, n):                       # AR(1) synoptic variability
        weather[t] = 0.72 * weather[t - 1] + rng.normal(0.0, 2.6)
    tmax = temp_mean + weather + 6.0 + rng.normal(0.0, 1.1, n)
    tmin = temp_mean + weather - 5.0 + rng.normal(0.0, 1.1, n)

    # --- precipitation: wet-day occurrence modulated by season ---------------
    wet_season = 0.20 + 0.12 * np.cos(2.0 * np.pi * (doy - (60 if regime == "rainfall" else 150)) / 365.25)
    wet = rng.random(n) < np.clip(wet_season, 0.03, 0.55)
    depth = rng.gamma(shape=0.75, scale=11.0 if regime == "rainfall" else 6.5, size=n)
    precip = np.where(wet, depth, 0.0)

    # --- inflow: baseflow recession, melt term, storm response ---------------
    if regime == "snowmelt":
        melt_drive = np.clip(temp_mean - 2.0, 0.0, None) * _seasonal(doy, peak_day=160, amplitude=0.5).clip(0.0)
        seasonal_component = 1.0 + 2.4 * melt_drive / (melt_drive.max() + 1e-9)
    else:
        seasonal_component = 1.0 + 0.55 * np.clip(
            _seasonal(doy, peak_day=60, amplitude=1.0), 0.0, None)

    # storm hydrograph: an exponentially decaying response to each rainfall day
    kernel = np.exp(-np.arange(12) / 3.2)
    kernel /= kernel.sum()
    storm = np.convolve(precip, kernel, mode="full")[:n]

    baseflow = np.zeros(n)
    for t in range(1, n):                       # AR(1) recession around the seasonal level
        baseflow[t] = 0.94 * baseflow[t - 1] + rng.normal(0.0, 0.08)

    inflow = mean_inflow * seasonal_component * np.exp(baseflow) * (
        1.0 + storm_response * storm / (storm.std() + 1e-9) * 0.25)
    inflow = np.clip(inflow, mean_inflow * 0.02, None)

    # --- storage and water-surface elevation ---------------------------------
    release = inflow.mean()
    storage = np.zeros(n)
    storage[0] = mean_inflow * 90.0
    for t in range(1, n):
        storage[t] = np.clip(storage[t - 1] + 0.6 * (inflow[t] - release),
                             mean_inflow * 20.0, mean_inflow * 220.0)
    elevation = ground_elev + 18.0 * (storage - storage.min()) / (np.ptp(storage) + 1e-9)

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "inflow": np.round(inflow, 2),
        "storage": np.round(storage, 1),
        "elevation": np.round(elevation, 2),
        "precip": np.round(precip, 2),
        "tmax": np.round(tmax, 2),
        "tmin": np.round(tmin, 2),
    })


def main() -> None:
    dates = pd.date_range(START, END, freq="D")
    align_dir = HERE / "align"
    meta_dir = HERE / "meta"
    align_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for offset, (rid, regime, mean_inflow, storm, lat, lon, elev, area) in enumerate(RESERVOIRS):
        rng = np.random.default_rng(SEED + offset)
        frame = _simulate(rng, dates, regime, mean_inflow, storm, elev)
        frame.to_csv(align_dir / f"{rid}.csv", index=False, lineterminator="\n")
        meta_rows.append({"NIDID": rid, "LATITUDE": lat, "LONGITUDE": lon,
                          "ELEV_M": elev, "SURFACE_AREA_KM2": area})
        print(f"[sample] {rid}: {len(frame)} days, "
              f"inflow {frame['inflow'].min():.1f}-{frame['inflow'].max():.1f}")

    pd.DataFrame(meta_rows).to_csv(
        meta_dir / "reservoir_latlon_elev_surface_area.csv", index=False, lineterminator="\n")

    for name, pool in (("sample_source", SOURCE_POOL), ("sample_target", TARGET_POOL)):
        (HERE / f"reservoirs_{name}.txt").write_text("\n".join(pool) + "\n", encoding="utf-8")

    print(f"[sample] wrote {len(RESERVOIRS)} records, the attribute table and the two pools")


if __name__ == "__main__":
    main()

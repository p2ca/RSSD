"""Reservoir-level NSE stability figures for the results section.

Both figures answer the same question — whether a model's advantage holds up reservoir by
reservoir rather than only on average — and differ only in which models they compare:

``role_stability``
    the LSTM baseline, the three reservoir-information variants and RSSD.
``alignment_stability``
    the three whole-feature alignment baselines and RSSD.

For each model, target reservoir and lead day, the NSE of the two single-regime transfer
scenarios that reach that target is averaged (mixed-source scenarios are excluded). Each
of the seven resulting daily values is assigned to one of the five NSE classes, and one
stacked bar reports how many lead days fall in each class. A bar therefore always totals
seven.

The numbers come from the ``per_reservoir_daily_r2_*.csv`` that
``python -m rssd.cli.evaluate`` writes, so every scenario in the figure has to have been
evaluated first. A run directory that holds several of them is ambiguous: the most recent
is used and named on stdout, and ``timestamp`` pins one instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rssd import paths
from rssd.config import MODEL_VARIANTS, eval_run_dir

__all__ = ["FIGURES", "NSE_CLASSES", "classify_nse", "build_figure"]

LEAD_DAYS = list(range(1, 8))

# The classes the manuscript groups reservoir-level NSE into, from worst to best.
NSE_CLASSES = [
    ("Unsatisfactory", "#d73027"),
    ("Acceptable", "#fc8d59"),
    ("Satisfactory", "#fee08b"),
    ("Good", "#91bfdb"),
    ("Very good", "#1a9850"),
]

# One panel per target group: the pool list it reads, and the two single-regime scenarios
# whose NSE is averaged before classification.
PANELS = [
    ("(a) Snowmelt-dominated targets", "snow_target_v2", ("snow2snow", "rain2snow")),
    ("(b) Rainfall-dominated targets", "rain_target_v2", ("rain2rain", "snow2rain")),
]

# variant key -> hatch. The RSSD bar is left unhatched so it reads as the focal series.
FIGURES = {
    "role_stability": {
        "variants": [("seq2seq_lstm", ""), ("attribute_informed_lstm", "/"),
                     ("identity_informed_lstm", "\\"), ("fully_informed_lstm", "x"),
                     ("rssd_lstm", "")],
        "filename": "role_nse_reservoir_stability",
    },
    "alignment_stability": {
        "variants": [("fully_informed_mmd", ""), ("fully_informed_coral", "/"),
                     ("fully_informed_dann", "\\"), ("rssd_lstm", "")],
        "filename": "alignment_nse_reservoir_stability",
    },
}


def classify_nse(value: float) -> str:
    """The manuscript's NSE class for one value."""
    if value > 0.75:
        return "Very good"
    if value > 0.65:
        return "Good"
    if value > 0.50:
        return "Satisfactory"
    if value > 0.40:
        return "Acceptable"
    return "Unsatisfactory"


def _read_pool(dataset_tag: str) -> list:
    """The reservoirs of one target pool, in the order the list file gives them."""
    list_file = paths.reservoir_list(dataset_tag)
    if not list_file.is_file():
        raise FileNotFoundError(f"reservoir list not found: {list_file}")
    return [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def resolve_daily_csv(scenario: str, variant: str, version: str,
                      timestamp: str = None) -> Path:
    """The per-lead-day NSE table to read for one scenario and variant.

    Evaluation runs accumulate in the run directory, so when more than one is present the
    most recent is taken and the choice is printed; ``timestamp`` selects a specific run.
    """
    run_dir = Path(eval_run_dir(scenario, variant, version))
    pattern = (f"per_reservoir_daily_r2_{timestamp}.csv" if timestamp
               else "per_reservoir_daily_r2_*.csv")
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no {pattern} in {run_dir}. Run\n"
            f"    python -m rssd.cli.evaluate --scenario {scenario} "
            f"--variant {variant} --version {version}")
    if len(matches) > 1:
        print(f"[FIGURE] {scenario}/{variant}: {len(matches)} evaluation runs present, "
              f"using {matches[-1].name}")
    return matches[-1]


def daily_nse(scenario: str, variant: str, version: str,
              timestamp: str = None) -> pd.DataFrame:
    """``reservoir`` x lead day NSE for one scenario and variant."""
    path = resolve_daily_csv(scenario, variant, version, timestamp)
    frame = pd.read_csv(path)
    columns = [f"r2_d{d}" for d in LEAD_DAYS]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing {missing}")
    return frame.set_index("reservoir")[columns]


def class_counts(variant: str, scenarios, reservoirs, version: str,
                 timestamp: str = None) -> np.ndarray:
    """``(reservoirs, classes)`` counts of lead days, after averaging over the scenarios."""
    tables = [daily_nse(scenario, variant, version, timestamp) for scenario in scenarios]

    counts = np.zeros((len(reservoirs), len(NSE_CLASSES)), dtype=int)
    class_index = {name: i for i, (name, _colour) in enumerate(NSE_CLASSES)}
    for row, reservoir in enumerate(reservoirs):
        for table in tables:
            if reservoir not in table.index:
                raise KeyError(f"{reservoir} missing from a {variant} evaluation table")
        averaged = np.mean([table.loc[reservoir].to_numpy(dtype=float) for table in tables],
                           axis=0)
        for value in averaged:
            counts[row, class_index[classify_nse(value)]] += 1
    return counts


def build_figure(figure: str, version: str = "v2", output_dir=None,
                 formats=("pdf", "png"), timestamp: str = None):
    """Draw one of :data:`FIGURES` and write it in each requested format."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if figure not in FIGURES:
        raise KeyError(f"unknown figure {figure!r}; known: {sorted(FIGURES)}")
    spec = FIGURES[figure]
    variants = spec["variants"]

    output_dir = Path(output_dir) if output_dir else paths.LOGS_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "hatch.linewidth": 0.45, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.4), sharey=True)

    # Most of each reservoir's slot goes to the bars; the hatching stays sparse so the
    # NSE-class colours remain the dominant encoding.
    group_span = 0.88
    bar_width = group_span / len(variants)
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2.0) * bar_width

    for ax, (title, pool_tag, scenarios) in zip(axes, PANELS):
        reservoirs = _read_pool(pool_tag)
        x = np.arange(len(reservoirs), dtype=float)

        for position, (variant, hatch) in enumerate(variants):
            counts = class_counts(variant, scenarios, reservoirs, version, timestamp)
            bottoms = np.zeros(len(reservoirs), dtype=float)
            edge_width = 0.65 if variant == "rssd_lstm" else 0.25
            for class_idx, (_name, colour) in enumerate(NSE_CLASSES):
                heights = counts[:, class_idx]
                ax.bar(x + offsets[position], heights, width=bar_width * 0.96,
                       bottom=bottoms, color=colour, edgecolor="#2b2b2b",
                       linewidth=edge_width, hatch=hatch, zorder=3)
                bottoms += heights

        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(reservoirs, rotation=35, ha="right")
        ax.set_ylim(0, len(LEAD_DAYS))
        ax.set_yticks(range(0, len(LEAD_DAYS) + 1))
        ax.grid(axis="y", linewidth=0.35, alpha=0.45, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Number of lead days")

    # Two legend rows below the panels: the model each bar is, in left-to-right order,
    # then the NSE classes. The model icons repeat the bar styling, with the hatch
    # doubled so it stays legible at icon size.
    model_handles = [
        Patch(facecolor="white", edgecolor="#2b2b2b",
              linewidth=0.65 if variant == "rssd_lstm" else 0.25,
              hatch=hatch * 2, label=MODEL_VARIANTS[variant]["label"])
        for variant, hatch in variants
    ]
    fig.legend(handles=model_handles, loc="lower center", bbox_to_anchor=(0.5, 0.100),
               ncol=min(len(variants), 3), frameon=False, handlelength=1.6,
               columnspacing=1.0)

    class_handles = [Patch(facecolor=colour, edgecolor="#2b2b2b", linewidth=0.25, label=name)
                     for name, colour in reversed(NSE_CLASSES)]
    fig.legend(handles=class_handles, loc="lower center", bbox_to_anchor=(0.5, 0.052),
               ncol=len(NSE_CLASSES), frameon=False, handlelength=1.6, columnspacing=1.2)

    fig.text(0.5, 0.022, "NSE class", ha="center", va="center", fontsize=7)
    fig.tight_layout(rect=(0.02, 0.185, 1.0, 0.995), w_pad=0.85)

    written = []
    for suffix in formats:
        out_path = output_dir / f"{spec['filename']}.{suffix}"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        written.append(out_path)
        print(f"[FIGURE] {out_path}")

    plt.close(fig)
    return written

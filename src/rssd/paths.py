"""Filesystem layout for the RSSD experiments.

Every path used by the package resolves through this module, so a checkout can be run
from any directory and on any machine. Three environment variables override the
defaults:

``RSSD_ROOT``
    Project root. Defaults to the repository root (two levels above this file).
``RSSD_DATA``
    Directory holding ``reservoirs_*.txt``, ``meta/`` and ``parsed/``.
    Defaults to ``<RSSD_ROOT>/data``.
``RSSD_LOGS``
    Directory holding training/evaluation run outputs.
    Defaults to ``<RSSD_ROOT>/logs``.

The reservoir records and the trained checkpoints are not distributed with the code,
so ``RSSD_DATA`` and ``RSSD_LOGS`` normally point at a local working copy:

.. code-block:: bash

    export RSSD_DATA=/path/to/reservoir_data
    export RSSD_LOGS=/path/to/runs

``RSSD_DATA`` can also point at the synthetic sample bundle shipped in ``data/sample``
(:data:`SAMPLE_DATA_DIR`), which has the same layout and lets the whole pipeline run
without the reservoir records.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "LOGS_DIR", "SAMPLE_DATA_DIR",
    "parsed_dir", "reservoir_list", "meta_dir", "align_dir",
    "train_log_dir", "eval_log_dir",
]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT: Path = _env_path("RSSD_ROOT", Path(__file__).resolve().parents[2])
DATA_DIR: Path = _env_path("RSSD_DATA", PROJECT_ROOT / "data")
LOGS_DIR: Path = _env_path("RSSD_LOGS", PROJECT_ROOT / "logs")

# The synthetic bundle shipped with the code, so the pipeline can be run without the
# reservoir records. It has the same layout as RSSD_DATA: align/, meta/, reservoirs_*.txt.
SAMPLE_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "sample"


def parsed_dir(dataset_tag: str) -> Path:
    """Directory of windowed tensors for one dataset tag, e.g. ``snow_source_v2``."""
    return DATA_DIR / "parsed" / dataset_tag


def reservoir_list(dataset_tag: str) -> Path:
    """The frozen reservoir list backing one dataset tag."""
    return DATA_DIR / f"reservoirs_{dataset_tag}.txt"


def meta_dir() -> Path:
    """Directory of static reservoir attribute tables."""
    return DATA_DIR / "meta"


def align_dir() -> Path:
    """Directory of per-reservoir aligned daily records (inflow + meteorology)."""
    return DATA_DIR / "align"


def train_log_dir(domain: str, run_group: str, version: str) -> Path:
    """``logs/train_<domain>/<run_group>/<version>`` — unchanged from the original layout."""
    return LOGS_DIR / f"train_{domain}" / run_group / version


def eval_log_dir(domain: str, run_group: str, version: str) -> Path:
    """``logs/eval_<domain>/<run_group>/<version>`` — unchanged from the original layout."""
    return LOGS_DIR / f"eval_{domain}" / run_group / version

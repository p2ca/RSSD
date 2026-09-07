"""Build a parsed dataset from aligned per-reservoir daily records.

    python -m rssd.cli.preprocess --dataset sample_source --role source
    python -m rssd.cli.preprocess --dataset sample_target --role target

With no ``--data-root`` the command reads the synthetic sample bundle shipped in
``data/sample/`` (see ``data/sample/README.md``), unless ``RSSD_DATA`` is set, in which
case that directory is used. Point ``--data-root`` at a directory holding your own
``align/<id>.csv`` records and ``reservoirs_<dataset tag>.txt`` pool lists to preprocess
those instead; the column contract is documented in :mod:`rssd.data.preprocess`.
"""

from __future__ import annotations

import argparse
import os
import sys

from rssd import paths
from rssd.data.preprocess import TARGET_HISTORY_YEARS, build_parsed_dataset

__all__ = ["main", "build_parser", "default_data_root"]


def default_data_root() -> str:
    """``RSSD_DATA`` when it is set, otherwise the sample bundle in the checkout."""
    return str(paths.DATA_DIR if os.environ.get("RSSD_DATA") else paths.SAMPLE_DATA_DIR)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True,
                   help="dataset tag to build, e.g. sample_source / snow_source_v2")
    p.add_argument("--role", default="source", choices=["source", "target"],
                   help="target pools keep only their most recent "
                        f"{TARGET_HISTORY_YEARS} years of record")
    p.add_argument("--data-root", default=None,
                   help=f"directory holding align/ and the pool lists (default: {default_data_root()})")
    p.add_argument("--reservoir-list", default=None,
                   help="pool list file (default: <data root>/reservoirs_<dataset tag>.txt)")
    p.add_argument("--output-dir", default=None,
                   help="write here instead of <data root>/parsed/<dataset tag>")
    p.add_argument("--input-days", type=int, default=30, help="input window length")
    p.add_argument("--forecast-days", type=int, default=7, help="forecast horizon")
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    data_root = args.data_root or default_data_root()
    print(f"[INFO] data root: {data_root}")

    return build_parsed_dataset(
        args.dataset,
        data_root=data_root,
        reservoir_list_file=args.reservoir_list,
        output_dir=args.output_dir,
        role=args.role,
        days_x=args.input_days,
        days_y=args.forecast_days,
    )


if __name__ == "__main__":
    main()

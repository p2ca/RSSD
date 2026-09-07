"""Draw the reservoir-level NSE stability figures of the results section.

    python -m rssd.cli.figures --figure role_stability
    python -m rssd.cli.figures --figure alignment_stability
    python -m rssd.cli.figures --all

Each figure reads the ``per_reservoir_daily_r2_*.csv`` files that
``python -m rssd.cli.evaluate`` writes, so the four single-regime scenarios
(``snow2snow``, ``rain2snow``, ``rain2rain``, ``snow2rain``) must have been evaluated for
every model the figure compares. Needs the optional ``figures`` extra:
``pip install -e ".[figures]"``.
"""

from __future__ import annotations

import argparse
import sys

from rssd.figures.nse_stability import FIGURES, build_figure

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--figure", choices=sorted(FIGURES),
                   help="which figure to draw")
    p.add_argument("--all", action="store_true", help="draw every figure")
    p.add_argument("--version", default="v2", help="version tag of the evaluation runs")
    p.add_argument("--output-dir", default=None,
                   help="write here instead of logs/plots")
    p.add_argument("--format", default="pdf,png",
                   help="comma-separated output formats (default: pdf,png)")
    p.add_argument("--timestamp", default=None,
                   help="read this evaluation run instead of the most recent one")
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    if not args.figure and not args.all:
        build_parser().error("give --figure or --all")

    names = sorted(FIGURES) if args.all else [args.figure]
    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())

    written = []
    for name in names:
        written += build_figure(name, version=args.version, output_dir=args.output_dir,
                                formats=formats, timestamp=args.timestamp)
    return written


if __name__ == "__main__":
    main()

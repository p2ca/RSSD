"""Train one source model.

    python -m rssd.cli.train --variant rssd_lstm --dataset snow_source_v2 --version v2

The variant selects the rung of the information-utilisation ladder; the configuration is
derived from the frozen defaults by :func:`rssd.config.build_train_cfg`, which applies the
variant profile and the experiment overrides. Checkpoints land in the project's usual
``logs/train_<domain>/<run group>/<version>/`` layout.
"""

from __future__ import annotations

import argparse
import json
import sys

from rssd.config import MODEL_VARIANTS, build_train_cfg, manifest_training_runs
from rssd.engine.train_loop import train_source_model

__all__ = ["main", "build_parser"]

# alignment baselines need an unlabelled target pool to align against
DEFAULT_ALIGN_TARGET = {"snow_source_v2": "rain_target_v2", "rain_source": "snow_target_v2",
                        "mixed_source_v2": "rain_target_v2",
                        "sample_source": "sample_target"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant",
                   help=f"model variant: {', '.join(sorted(MODEL_VARIANTS))}")
    p.add_argument("--dataset",
                   help="source dataset tag, e.g. snow_source_v2 / rain_source / mixed_source_v2")
    p.add_argument("--version", default="v2", help="version tag for the run directory")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--align-target", default=None,
                   help="target dataset tag the alignment baselines align against")
    p.add_argument("--epochs", type=int, default=None, help="cap on training epochs")
    p.add_argument("--device", default=None, help="cuda:0 / cpu (default: cuda when available)")
    p.add_argument("--output-dir", default=None,
                   help="write checkpoints here instead of the standard logs/train_* location")
    p.add_argument("--no-save", action="store_true", help="run without writing checkpoints")
    p.add_argument("--print-config", action="store_true",
                   help="print the resolved configuration and exit without training")
    p.add_argument("--list-runs", action="store_true",
                   help="print the training commands declared in configs/experiments.yaml")
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    if args.list_runs:
        for run in manifest_training_runs():
            target = f" --align-target {run['align_target']}" if run["align_target"] else ""
            print(f"python -m rssd.cli.train --variant {run['variant']} "
                  f"--dataset {run['dataset']} --version {run['version']}{target}"
                  f"    # {run['name']}, {run['pool']} source")
        return None

    if not args.variant or not args.dataset:
        build_parser().error("--variant and --dataset are required unless --list-runs is given")

    overrides = {}

    # only the alignment baselines consult a target pool; leave the field untouched otherwise
    needs_target = MODEL_VARIANTS.get(args.variant, {}).get("align_method", "none") != "none"
    align_target = args.align_target or (DEFAULT_ALIGN_TARGET.get(args.dataset)
                                         if needs_target else None)
    cfg = build_train_cfg(variant=args.variant, dataset_tag=args.dataset,
                          version_tag=args.version, seed=args.seed,
                          target_dataset_tag=align_target, overrides=overrides or None)

    if args.print_config:
        print(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
        return cfg

    result = train_source_model(cfg, device=args.device, output_dir=args.output_dir,
                                max_epochs=args.epochs, save_bundles=not args.no_save)
    print(f"\nbest epoch {result['best_epoch']} | {result['best_metric_name']} "
          f"{result['best_metric']:.6f} | weights: {result['final_state_tag']}")
    return result


if __name__ == "__main__":
    main()

"""Evaluate a trained checkpoint on a target reservoir set.

    python -m rssd.cli.evaluate --scenario snow2rain --variant rssd_lstm

The command runs the full evaluation path: it reads the architecture back
out of the checkpoint, rebuilds the model for the target node set, restores or initialises
the per-reservoir embedding, adapts on the target's own recent history, and writes the same
structured outputs (``per_reservoir_r2_*.csv``, ``metrics_summary_*.json``, ...) into the
same directory layout the project has always used.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime

import pandas as pd
import torch

from rssd import paths
from rssd.config import (SCENARIOS, EvalConfig, MODEL_VARIANTS, checkpoint_path,
                         eval_run_dir, load_manifest)
from rssd.data import datasets, static_attrs
from rssd.data.scalers import build_local_y_inverse_tensors
from rssd.engine.evaluator import evaluate_model
from rssd.engine.finetune import finetune_on_target
from rssd.io.bundle import _load_state_dict_any
from rssd.io.summary import _float_or_none, _jsonify, _metrics_summary_dict, _parse_ckpt_ts
from rssd.models import builder
from rssd.models.checkpoint import load_checkpoint_config
from rssd.utils import TrainingLogger, collate_zip, seed_everything

__all__ = ["run_evaluation", "main"]


def _resolve_variant(variant: str) -> dict:
    if variant in MODEL_VARIANTS:
        return MODEL_VARIANTS[variant]
    for spec in MODEL_VARIANTS.values():
        if spec["exp_name"] == variant:
            return spec
    raise KeyError(f"Unknown model variant {variant!r}. Known: {sorted(MODEL_VARIANTS)}")


def _checkpoint_version_tag(bundle: dict, fallback: str) -> str:
    tag = str(bundle.get("user_cfg", {}).get("experiment", {}).get("VERSION_TAG", "") or "")
    return tag or fallback


def run_evaluation(scenario: str, variant: str, version: str = "v2", *, ckpt=None,
                   cfg: EvalConfig = None, device=None, output_dir=None, write_outputs=True):
    """Run one scenario and return the metrics plus the paths written."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario!r}. Known: {sorted(SCENARIOS)}")
    spec = _resolve_variant(variant)
    scn = SCENARIOS[scenario]
    cfg = cfg or EvalConfig()
    device = builder.resolve_device(device)

    ckpt = str(ckpt) if ckpt else str(checkpoint_path(variant, scn["source_domain"], version))
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # 1. architecture is dictated by the checkpoint, never re-specified here
    ckpt_cfg, bundle = load_checkpoint_config(ckpt, scn["source_tag"], cfg.scaler_type)
    print(f"[EVAL] {scenario} | {spec['label']} | {ckpt_cfg.describe()}")

    # 2. target data
    # seed_everything (not the bare seed setter) is what the original runs used: it also
    # pins cuDNN to deterministic kernels, which is why those runs are bit-reproducible
    seed_everything(cfg.seed)
    ds = datasets.load_parsed_dataset(scn["eval_tag"], cfg.scaler_type)
    datasets.assert_target_scaler_consistency(ds)
    _, _, _, _ = datasets.compute_train_scale_stats(ds)   # fail fast on an inconsistent scaler

    support_dataset, test_dataset = datasets.build_target_datasets(ds)
    pin_memory = (torch.cuda.is_available() if cfg.eval_pin_memory is None
                  else bool(cfg.eval_pin_memory))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_zip,
        num_workers=cfg.eval_num_workers, pin_memory=pin_memory)

    # 3. model sized for the target node set
    model = builder.build_model_from_checkpoint_config(
        ckpt_cfg, input_dim=int(ds.X_train.shape[-1]), pred_len=ds.pred_len,
        num_nodes=ds.num_nodes, device=device)

    node_sets = builder.compare_node_sets(ds.reservoir_names_in_node_order,
                                          ckpt_cfg.train_reservoir_names_in_node_order)
    uses_static = ckpt_cfg.use_res_static or ckpt_cfg.use_meta_only_static
    static_info = None
    if uses_static:
        # attributes are normalised with the SOURCE reservoirs' statistics, so the
        # checkpoint sees them on the scale it was trained on
        static_info = static_attrs.build_static_matrix(
            ds.reservoir_names_in_node_order, ckpt_cfg.meta_feature_names,
            reference_names=ckpt_cfg.train_reservoir_names_in_node_order, clip=None)
        builder.attach_static_attributes(model, static_info["normalized"],
                                         use_res_static=ckpt_cfg.use_res_static,
                                         use_meta_only_static=ckpt_cfg.use_meta_only_static,
                                         device=device)
        print(f"[EVAL] static drift under source normalisation: "
              f"abs-p95={static_info['abs_p95']:.3f} abs-max={static_info['abs_max']:.3f}")

    state_dict, ckpt_meta = _load_state_dict_any(ckpt)
    report = builder.load_source_weights(model, state_dict, node_sets=node_sets,
                                         backbone=ckpt_cfg.backbone)
    print(f"[EVAL] weights loaded={report['loaded']} skipped={len(report['skipped'])} "
          f"same_nodes={node_sets['same_nodes']} same_order={node_sets['same_order']}")

    # cross-dataset: re-apply the clipped attribute matrix after the load
    if ckpt_cfg.use_res_static and not node_sets["same_nodes"]:
        if cfg.cross_static_fatal_p95 > 0.0 and static_info["abs_p95"] > cfg.cross_static_fatal_p95:
            raise RuntimeError("[CROSSSET][FATAL] source-normalised target attributes are too far "
                               f"out of range (abs-p95={static_info['abs_p95']:.3f} > "
                               f"{cfg.cross_static_fatal_p95:.3f}).")
        clipped = static_attrs.apply_norm(static_info["raw"], static_info["med"],
                                          static_info["iqr"], clip=cfg.cross_static_norm_clip)
        builder.attach_static_attributes(model, clipped, use_res_static=True,
                                         use_meta_only_static=False, device=device)

    emb_method = builder.initialize_target_embedding(
        model, state_dict.get("reservoir_emb.weight", None), node_sets=node_sets,
        target_names=ds.reservoir_names_in_node_order,
        source_names=ckpt_cfg.train_reservoir_names_in_node_order,
        feature_names=ckpt_cfg.meta_feature_names, use_static=uses_static,
        use_meta_emb_init=cfg.use_meta_emb_init, ridge_lambda=cfg.meta_emb_ridge_lambda,
        blend_alpha=cfg.meta_emb_blend_alpha, norm_clip=cfg.cross_static_norm_clip,
        fatal_p95=cfg.cross_static_fatal_p95)
    print(f"[EVAL] reservoir embedding: {emb_method}")

    inv_pack_y = build_local_y_inverse_tensors(ds.scaler_data, ds.reservoir_names_in_node_order,
                                               device)

    # 4. target-history adaptation
    is_domain_shift = (scn["source_tag"] != scn["eval_tag"]) or (not node_sets["same_nodes"])
    do_finetune = cfg.enable_finetune and (not cfg.finetune_only_if_domain_shift or is_domain_shift)
    if do_finetune:
        print(f"[FINETUNE] mode={cfg.finetune_mode} domain_shift={is_domain_shift}")
        model = finetune_on_target(
            model=model, train_dataset_full=support_dataset, collate_fn=collate_zip,
            device=device, inv_pack_y=inv_pack_y, max_epochs=cfg.finetune_max_epochs,
            lr=cfg.finetune_lr, weight_decay=cfg.finetune_weight_decay,
            grad_clip=cfg.finetune_grad_clip, val_frac=cfg.finetune_val_frac,
            patience=cfg.finetune_patience, seed=cfg.finetune_seed,
            mode=cfg.finetune_mode, eval_batch_size=cfg.eval_batch_size,
            num_workers=cfg.eval_num_workers, pin_memory=pin_memory,
            cache_to_device=cfg.finetune_cache_to_device,
            train_max_batches=cfg.finetune_train_max_batches,
            val_max_batches=cfg.finetune_val_max_batches,
            clamp_pred_to_fr=cfg.clamp_pred_to_fr, min_delta=cfg.finetune_min_delta)
    else:
        print(f"[FINETUNE] disabled | domain_shift={is_domain_shift}")

    # 5. evaluation
    overall_r2, daily_r2_scores, reservoir_r2_scores, reservoir_r2_daily = evaluate_model(
        model, test_loader, ds.encode_map, ds.scaler_data, device, inv_pack_y,
        ds.reservoir_names_in_node_order, clamp_pred_to_fr=cfg.clamp_pred_to_fr,
        exclude_reservoirs=cfg.exclude_reservoirs, monitor_reservoirs=cfg.monitor_reservoirs,
        overall_r2_mode=cfg.overall_r2_mode)

    result = {
        "scenario": scenario, "variant": variant, "label": spec["label"],
        "checkpoint": ckpt, "overall_r2": overall_r2, "daily_r2": list(daily_r2_scores),
        "per_reservoir": dict(reservoir_r2_scores), "per_reservoir_daily": dict(reservoir_r2_daily),
        "avg_nse": float(sum(reservoir_r2_scores.values()) / max(len(reservoir_r2_scores), 1)),
        "embedding_init": emb_method, "is_domain_shift": is_domain_shift,
        "finetuned": bool(do_finetune),
    }
    print(f"[RESULT] {scenario} {spec['label']}: avg-NSE = {result['avg_nse']:.4f}")

    if not write_outputs:
        return result

    # 6. structured outputs, in the project's directory layout
    version_dir = _checkpoint_version_tag(bundle, version)
    log_dir = str(output_dir) if output_dir else str(eval_run_dir(scenario, variant, version_dir))
    src_domain, tgt_domain = scn["source_domain"], scn["target_domain"]
    model_name = f"eval_{src_domain}2{tgt_domain}_{ckpt_cfg.exp_name}"
    logger = TrainingLogger(model_name, cfg.scaler_type, log_dir_override=log_dir,
                            timestamp_override=datetime.now().strftime("%Y%m%d%H%M"))

    eval_meta = {
        "eval_dataset_tag": scn["eval_tag"], "source_dataset_tag": scn["source_tag"],
        "scaler_type": cfg.scaler_type, "source_ckpt_path": ckpt,
        "source_ckpt_ts": _parse_ckpt_ts(ckpt), "model_name": model_name,
        "exp_name": ckpt_cfg.exp_name, "model_variant": ckpt_cfg.model_variant,
        "manuscript_name": spec["label"],
        "latent_mode": ckpt_cfg.latent_mode, "use_latent_proj": ckpt_cfg.use_latent_proj,
        "use_err_head": ckpt_cfg.use_err_head, "use_darsd": ckpt_cfg.use_darsd,
        "lcib_k": ckpt_cfg.lcib_k, "darsd_mode": ckpt_cfg.darsd_mode,
        "emb_dropout_p": ckpt_cfg.emb_dropout_p, "use_res_static": ckpt_cfg.use_res_static,
        "res_static_mode": ckpt_cfg.res_static_mode,
        "use_meta_only_static": ckpt_cfg.use_meta_only_static,
        "meta_only_static_dim": ckpt_cfg.meta_only_static_dim,
        "meta_feature_names": list(ckpt_cfg.meta_feature_names),
        "meta_feature_strengths": [float(v) for v in ckpt_cfg.meta_feature_strengths],
        "res_static_dim": ckpt_cfg.res_static_dim, "num_nodes_eval": ds.num_nodes,
        "target_support_samples": int(len(support_dataset)),
        "reservoir_names_in_node_order": list(ds.reservoir_names_in_node_order),
        "ckpt_meta": ckpt_meta,
        "use_meta_emb_init": cfg.use_meta_emb_init,
        "meta_emb_init_method": emb_method,
        "meta_emb_blend_alpha": cfg.meta_emb_blend_alpha,
        "meta_emb_ridge_lambda": cfg.meta_emb_ridge_lambda,
        "enable_finetune": bool(do_finetune), "finetune_mode": cfg.finetune_mode,
        "static_drift": {k: static_info[k] for k in ("abs_max", "abs_p95", "abs_p99")}
                        if static_info else None,
        "weight_load_report": {k: v for k, v in report.items() if k != "skipped"},
        "eval_cfg": copy.deepcopy(cfg.as_metadata()),
    }

    written = {}
    ts = logger.timestamp

    info = ["\n\nFinal Evaluation Results:", f"Overall R2 Score: {overall_r2:.4f}",
            f"Daily R2 Scores: {[f'{x:.4f}' for x in daily_r2_scores]}",
            "Per-Reservoir R2 Scores (flatten over horizon):"]
    info += [f"  {name}: {val:.4f}" for name, val in sorted(reservoir_r2_scores.items())]
    info.append("Per-Reservoir Daily R2 Scores:")
    for name, daily in sorted(reservoir_r2_daily.items()):
        info.append(f"  {name}: " + ", ".join(f"d{i+1}={v:.4f}" for i, v in enumerate(daily)))
    written["results"] = os.path.join(log_dir, f"results_{ts}.txt")
    with open(written["results"], "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")

    written["metadata"] = os.path.join(log_dir, f"eval_metadata_{ts}.json")
    with open(written["metadata"], "w", encoding="utf-8") as f:
        json.dump(_jsonify(eval_meta), f, indent=2, ensure_ascii=False)

    metrics_summary = _metrics_summary_dict(
        eval_meta=eval_meta, logger=logger, overall_r2=overall_r2,
        daily_r2_scores=daily_r2_scores, reservoir_r2_scores=reservoir_r2_scores,
        reservoir_r2_daily=reservoir_r2_daily, is_domain_shift=is_domain_shift)
    written["metrics_summary"] = os.path.join(log_dir, f"metrics_summary_{ts}.json")
    with open(written["metrics_summary"], "w", encoding="utf-8") as f:
        json.dump(_jsonify(metrics_summary), f, indent=2, ensure_ascii=False)

    written["summary_row"] = os.path.join(log_dir, f"summary_row_{ts}.csv")
    pd.DataFrame([metrics_summary]).to_csv(written["summary_row"], index=False,
                                           encoding="utf-8-sig")

    base_row = {
        "timestamp": ts, "exp_name": eval_meta["exp_name"],
        "model_variant": eval_meta["model_variant"],
        "source_dataset_tag": eval_meta["source_dataset_tag"],
        "eval_dataset_tag": eval_meta["eval_dataset_tag"],
    }
    rows = [dict(base_row, reservoir=name, r2_flat=_float_or_none(val))
            for name, val in sorted(reservoir_r2_scores.items())]
    written["per_reservoir"] = os.path.join(log_dir, f"per_reservoir_r2_{ts}.csv")
    pd.DataFrame(rows).to_csv(written["per_reservoir"], index=False, encoding="utf-8-sig")

    daily_rows = []
    for name, daily in sorted(reservoir_r2_daily.items()):
        row = dict(base_row, reservoir=name)
        row.update({f"r2_d{i}": _float_or_none(v) for i, v in enumerate(daily, start=1)})
        daily_rows.append(row)
    written["per_reservoir_daily"] = os.path.join(log_dir, f"per_reservoir_daily_r2_{ts}.csv")
    pd.DataFrame(daily_rows).to_csv(written["per_reservoir_daily"], index=False,
                                    encoding="utf-8-sig")

    for path in written.values():
        print("[EXPORT] saved:", path)
    result["written"] = written
    result["log_dir"] = log_dir
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", choices=sorted(SCENARIOS))
    p.add_argument("--variant",
                   help=f"model variant: {', '.join(sorted(MODEL_VARIANTS))}")
    p.add_argument("--version", default="v2", help="checkpoint version directory (default: v2)")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path, overriding --version")
    p.add_argument("--device", default=None, help="cuda:0 / cpu (default: cuda when available)")
    p.add_argument("--output-dir", default=None,
                   help="write outputs here instead of the standard logs/eval_* location")
    p.add_argument("--no-finetune", action="store_true",
                   help="evaluate the source checkpoint without target-history adaptation")
    p.add_argument("--val-frac", type=float, default=None,
                   help="share of the support windows held out for adaptation early "
                        "stopping (default: 0.10)")
    p.add_argument("--finetune-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-outputs", action="store_true", help="compute metrics without writing files")
    p.add_argument("--list-runs", action="store_true",
                   help="print the evaluation commands declared in configs/experiments.yaml")
    return p


def main(argv=None):
    # the run prints hydrological symbols; keep them readable on a non-UTF-8 console
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)

    if args.list_runs:
        manifest = load_manifest()
        version = str(manifest.get("version", "v2"))
        for variant in manifest["variants"]:
            for scenario in manifest["scenarios"]:
                print(f"python -m rssd.cli.evaluate --scenario {scenario['key']} "
                      f"--variant {variant['key']} --version {version}"
                      f"    # {variant['name']}, {scenario['kind']}")
        return None

    if not args.scenario or not args.variant:
        build_parser().error("--scenario and --variant are required unless --list-runs is given")

    cfg = EvalConfig()
    if args.no_finetune:
        cfg.enable_finetune = False
    if args.finetune_epochs is not None:
        cfg.finetune_max_epochs = args.finetune_epochs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.val_frac is not None:
        cfg.finetune_val_frac = args.val_frac

    result = run_evaluation(args.scenario, args.variant, args.version, ckpt=args.ckpt, cfg=cfg,
                            device=args.device, output_dir=args.output_dir,
                            write_outputs=not args.no_outputs)
    print(f"\navg-NSE {result['avg_nse']:.4f} over {len(result['per_reservoir'])} reservoirs")
    return result


if __name__ == "__main__":
    main()

"""Training a source model end to end.

One run is fully described by a single configuration dictionary, built by
:func:`rssd.config.build_train_cfg`.

The loop selects on the validation loss, keeps the best state on CPU, and stops once the
patience counter runs out. It writes ``best_bundle.pt``, which carries the architecture
evaluation reads back out of.
"""

from __future__ import annotations

import copy
import os
from datetime import datetime

import torch
import torch.nn as nn

from rssd.config import train_run_dir
from rssd.data import datasets, static_attrs
from rssd.data.scalers import build_local_y_inverse_tensors
from rssd.engine.trainer import evaluate_model, run_epoch
from rssd.models import builder
from rssd.objectives.schedules import get_mmd_weight
from rssd.utils import TrainingLogger, seed_everything

__all__ = ["train_source_model"]


def _align_weight_max(cfg: dict) -> float:
    method = str(cfg["adaptation"]["ALIGN_METHOD"])
    return float({"mmd": cfg["adaptation"]["MMD_WEIGHT_MAX"],
                  "coral": cfg["adaptation"]["CORAL_WEIGHT_MAX"],
                  "dann": cfg["adaptation"]["DANN_LAMBDA_MAX"]}.get(method, 0.0))


def _bundle_config(cfg: dict, input_dim: int, pred_len: int) -> dict:
    """The architecture record a checkpoint carries; evaluation rebuilds the model from it."""
    m = cfg["model"]
    return {
        "input_dim": int(input_dim),
        "hidden_dim": int(m["HIDDEN_DIM"]),
        "output_dim": 1,
        "num_layers": int(m["NUM_LAYERS"]),
        "pred_len": int(pred_len),
        "dropout": float(m["DROPOUT"]),
        "use_reservoir_emb": bool(m["use_reservoir_emb"]),
        "reservoir_emb_dim": int(m["res_emb_dim"]),
        "use_res_static": bool(m["use_res_static"]),
        "use_meta_only_static": bool(m["use_meta_only_static"]),
        "meta_only_static_dim": int(m["meta_only_static_dim"]),
        "meta_feature_names": list(m["meta_feature_names"]),
        "res_static_dim": int(m["res_static_dim"]),
        "latent_mode": str(m["latent_mode"]),
        "use_latent_proj": bool(m["use_latent_proj"]),
        "use_darsd": bool(m["use_darsd"]),
        "lcib_k": int(m["lcib_k"]),
        "emb_dropout_p": float(m.get("emb_dropout_p", 0.0)),
        "backbone": str(m.get("BACKBONE", "lstm")),
        "n_heads": int(m.get("N_HEADS", 8)),
        "tf_layers": int(m.get("TF_LAYERS", 2)),
        "tf_ff_mult": int(m.get("TF_FF_MULT", 4)),
        "tin": int(m.get("TIN", 30)),
    }


def _bundle_meta(cfg: dict, ds, model_name: str) -> dict:
    return {
        "run_tag": str(cfg["experiment"]["RUN_TAG"]),
        "dataset_tag": str(cfg["experiment"]["DATASET_TAG"]),
        "scaler_type": str(cfg["experiment"]["SCALER_TYPE"]),
        "model_name": str(model_name),
        "exp_name": str(cfg["ablation"]["EXP_NAME"]),
        "model_variant": str(cfg["ablation"]["MODEL_VARIANT"]),
        "target_dataset_tag": str(cfg["adaptation"]["TARGET_DATASET_TAG"]),
        "use_mmd": bool(cfg["adaptation"]["USE_MMD"]),
        "mmd_weight_max": float(cfg["adaptation"]["MMD_WEIGHT_MAX"]),
        "align_method": str(cfg["adaptation"]["ALIGN_METHOD"]),
        "align_weight_max": _align_weight_max(cfg),
        "train_reservoir_names_in_node_order": list(ds.reservoir_names_in_node_order),
    }


def train_source_model(cfg: dict, *, device=None, output_dir=None, max_epochs=None,
                       save_bundles: bool = True):
    """Train one source model and return the run's outcome.

    Returns a dictionary with the best epoch and metric, the loss history, the paths
    written, and the trained model.
    """
    device = builder.resolve_device(device)
    exp = cfg["experiment"]
    m, obj, tr, dcfg, adapt = (cfg["model"], cfg["objective"], cfg["train"], cfg["data"],
                               cfg["adaptation"])
    align_method = str(adapt["ALIGN_METHOD"])
    use_align = align_method != "none"

    seed_everything(int(exp["SEED"]))

    # ---------------------------------------------------------------- data
    ds = datasets.load_parsed_dataset(str(exp["DATASET_TAG"]), str(exp["SCALER_TYPE"]))
    if str(exp["RUN_TAG"]) == "AUTO":
        # the run tag records which reservoirs the run actually trained on
        exp["RUN_TAG"] = ds.auto_run_tag()
    scale_arr, _scale_dict, _var_arr, _y_train_orig = datasets.compute_train_scale_stats(ds)

    train_dataset, val_dataset, test_dataset = datasets.build_datasets(ds)

    pin_memory = (torch.cuda.is_available() if dcfg["DL_PIN_MEMORY"] is None
                  else bool(dcfg["DL_PIN_MEMORY"]))
    num_workers = int(dcfg["DL_NUM_WORKERS"])
    loaders = datasets.build_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=int(dcfg["BATCH_SIZE"]), seed=int(exp["SEED"]),
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=int(dcfg["DL_PREFETCH_FACTOR"]) if num_workers > 0 else None)

    target_loader = None
    if use_align:
        target_loader, target_dataset = datasets.build_alignment_loader(
            str(adapt["TARGET_DATASET_TAG"]), scaler_type=str(exp["SCALER_TYPE"]),
            batch_size=int(dcfg["BATCH_SIZE"]), seed=int(exp["SEED"]),
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=int(dcfg["DL_PREFETCH_FACTOR"]) if num_workers > 0 else None)
        print(f"[ALIGN:{align_method}] target support = {adapt['TARGET_DATASET_TAG']} "
              f"| n={len(target_dataset)}")

    print(f"[DATA] {exp['DATASET_TAG']} | nodes={ds.num_nodes} pred_len={ds.pred_len} "
          f"| train={len(loaders['train_dataset'])} val={len(val_dataset)} test={len(test_dataset)}")

    # ---------------------------------------------------------------- model
    model = builder.build_model(
        input_dim=int(ds.X_train.shape[-1]), hidden_dim=int(m["HIDDEN_DIM"]),
        num_layers=int(m["NUM_LAYERS"]), pred_len=ds.pred_len, dropout=float(m["DROPOUT"]),
        latent_mode=str(m["latent_mode"]), use_latent_proj=bool(m["use_latent_proj"]),
        num_reservoirs=ds.num_nodes,
        use_reservoir_emb=bool(m["use_reservoir_emb"]),
        reservoir_emb_dim=int(m["res_emb_dim"]),
        emb_dropout_p=float(m.get("emb_dropout_p", 0.0)),
        use_res_static=bool(m["use_res_static"]), res_static_dim=int(m["res_static_dim"]),
        use_meta_only_static=bool(m["use_meta_only_static"]),
        meta_only_static_dim=int(m["meta_only_static_dim"]),
        use_darsd=bool(m["use_darsd"]), lcib_k=int(m["lcib_k"]),
        backbone=str(m.get("BACKBONE", "lstm")),
        n_heads=int(m.get("N_HEADS", 8)), tf_layers=int(m.get("TF_LAYERS", 2)),
        tf_ff_mult=int(m.get("TF_FF_MULT", 4)), tin=int(m.get("TIN", 30)), device=device)

    res_static = meta_static = None
    if m["use_res_static"] or m["use_meta_only_static"]:
        # source training normalises the attributes with its own reservoirs' statistics
        info = static_attrs.build_static_matrix(ds.reservoir_names_in_node_order,
                                                m["meta_feature_names"])
        builder.attach_static_attributes(model, info["normalized"],
                                         use_res_static=bool(m["use_res_static"]),
                                         use_meta_only_static=bool(m["use_meta_only_static"]),
                                         device=device)
        tensor = torch.tensor(info["normalized"], dtype=torch.float32)
        res_static = tensor if m["use_res_static"] else None
        meta_static = tensor if m["use_meta_only_static"] else None

    discriminator = None
    if align_method == "dann":
        discriminator = builder.build_domain_discriminator(
            int(m["HIDDEN_DIM"]), int(adapt["DANN_DISC_HIDDEN"]), device=device)

    optimizer, scheduler = builder.build_optimizer(
        model, lr=float(tr["LR"]), weight_decay=float(tr["WEIGHT_DECAY"]),
        discriminator=discriminator, disc_lr=float(adapt["DANN_DISC_LR"]))

    # ---------------------------------------------------------------- run bookkeeping
    domain = str(exp["DATASET_TAG"]).split("_")[0]
    model_name = f"train_{domain}_{cfg['ablation']['EXP_NAME']}"
    log_dir = str(output_dir) if output_dir else str(train_run_dir(cfg))
    logger = TrainingLogger(model_name, str(exp["SCALER_TYPE"]), log_dir_override=log_dir,
                            timestamp_override=datetime.now().strftime("%Y%m%d%H%M"))
    best_path = os.path.join(logger.log_dir, "best_bundle.pt")

    criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")
    inv_pack_y = build_local_y_inverse_tensors(ds.scaler_data,
                                               ds.reservoir_names_in_node_order, device)
    y_scale_t = torch.tensor(scale_arr, device=device, dtype=torch.float32)

    epoch_kwargs = dict(
        clamp_pred_to_fr=bool(obj["CLAMP_PRED_TO_FR"]),
        grad_clip_norm=float(tr["GRAD_CLIP_NORM"]),
        DARSD_WEIGHT=float(obj["DARSD_WEIGHT"]),
        mmd_kernel_num=int(adapt["MMD_KERNEL_NUM"]),
        mmd_kernel_mul=float(adapt["MMD_KERNEL_MUL"]),
        mmd_normalize_latent=bool(adapt["MMD_NORMALIZE_LATENT"]),
        mmd_max_samples_per_domain=int(adapt["MMD_MAX_SAMPLES_PER_DOMAIN"]),
    )

    max_epochs = int(max_epochs or tr["MAX_EPOCHS"])
    best_metric, best_epoch, best_state_cpu, best_val_loss = 1e18, -1, None, None
    best_metric_name = "val_loss"
    best_bundle, no_improve = None, 0
    history = []

    # ---------------------------------------------------------------- epochs
    for epoch in range(1, max_epochs + 1):
        lambda_align = get_mmd_weight(
            epoch, w_max=_align_weight_max(cfg),
            warmup=int(adapt["MMD_WARMUP_EPOCHS"]),
            ramp=int(adapt["MMD_RAMP_EPOCHS"])) if use_align else 0.0

        train_loss, train_mae = run_epoch(
            model, loaders["train"], criterion, inv_pack_y, y_scale_t,
            optimizer=optimizer, train=True, device=device, epoch=epoch,
            target_loader=target_loader, align_method=align_method,
            align_weight=lambda_align, domain_discriminator=discriminator,
            mmd_weight=lambda_align, **epoch_kwargs)

        val_loss, val_mae = run_epoch(
            model, loaders["val"], criterion, inv_pack_y, y_scale_t,
            optimizer=None, train=False, device=device, epoch=epoch,
            target_loader=None, align_method=align_method,
            align_weight=0.0, domain_discriminator=discriminator,
            mmd_weight=0.0, **epoch_kwargs)

        # the manuscript retains the checkpoint with the lowest validation loss
        selection_metric, selection_metric_name = float(val_loss), "val_loss"

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(selection_metric)

        print(f"[E{epoch:03d}] train_loss={train_loss:.6f} train_normMAE_mean={train_mae:.6f} | "
              f"val_loss={val_loss:.6f} val_normMAE_mean={val_mae:.6f} | "
              f"select={selection_metric_name}:{selection_metric:.6f} | lr={current_lr:.6g} | "
              f"align={align_method} lambda_align={float(lambda_align):.6f}")
        history.append({"epoch": epoch, "train_loss": float(train_loss),
                        "train_mae": float(train_mae), "val_loss": float(val_loss),
                        "val_mae": float(val_mae), "lr": float(current_lr)})

        if selection_metric < best_metric:
            best_metric, best_metric_name = selection_metric, selection_metric_name
            best_epoch, best_val_loss, no_improve = epoch, val_loss, 0

            best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_bundle = dict(
                state_dict=best_state_cpu, best_epoch=int(best_epoch),
                best_metric=float(best_metric),
                config=_bundle_config(cfg, int(ds.X_train.shape[-1]), ds.pred_len),
                user_cfg=copy.deepcopy(cfg),
                train_res_static_cpu=(res_static.detach().cpu() if res_static is not None else None),
                train_meta_static_cpu=(meta_static.detach().cpu() if meta_static is not None else None),
                **_bundle_meta(cfg, ds, model_name))
            if save_bundles:
                torch.save(best_bundle, best_path)
                print(f"[CKPT] saved best bundle at epoch {epoch} -> {best_path}")

            logger.save_checkpoint(model, optimizer, epoch, val_mae)
        else:
            no_improve += 1

        if no_improve >= int(tr["EARLY_STOP_PATIENCE"]):
            print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, "
                  f"best {best_metric_name}={best_metric:.6f})")
            break

    print(f"[BEST] epoch={best_epoch} best {best_metric_name}={best_metric:.6f}")

    # ---------------------------------------------------------------- finalize
    final_state_cpu, final_state_tag = best_state_cpu, "best"
    written = {}

    if final_state_cpu is not None:
        model.load_state_dict({k: t.to(device) for k, t in final_state_cpu.items()})
        print(f"[CKPT] final eval weights = {final_state_tag}")

    if best_bundle is not None and save_bundles:
        torch.save(best_bundle, best_path)
        written["best_bundle"] = best_path
        print("[CKPT] final best bundle written ->", best_path)

    return {
        "model": model, "cfg": cfg, "log_dir": logger.log_dir, "written": written,
        "best_epoch": best_epoch, "best_metric": float(best_metric),
        "best_metric_name": best_metric_name,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "final_state_tag": final_state_tag, "history": history,
        "dataset": ds, "loaders": loaders, "inv_pack_y": inv_pack_y,
        "criterion": criterion, "y_scale_t": y_scale_t,
    }

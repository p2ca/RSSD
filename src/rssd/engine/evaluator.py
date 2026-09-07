"""Evaluating a trained model on a target reservoir set.

``evaluate_model`` runs the target test block through the model, inverts the per-reservoir
scaling back to physical inflow units, and returns the pooled and per-reservoir skill. The
prediction clamp, the reservoir exclude and monitor lists, and whether the exported overall
R2 is pooled (micro) or reservoir-averaged (macro) are all keyword arguments.

Metrics are reported in both the scaled and the physical space, per lead day and per
reservoir, and a persistence forecast is scored alongside the model as a reference.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import r2_score

from rssd.common import _squeeze_pred_tensor, link_pred_to_scaled
from rssd.data.reservoirs import _build_idx_to_reservoir
from rssd.data.scalers import debug_scaled_impossibility
from rssd.metrics import _per_reservoir_r2, _per_reservoir_r2_daily
from rssd.utils import inverse_transform_predictions

__all__ = ["evaluate_model"]


def evaluate_model(model, test_loader, encode_map, scaler_data, device, inv_pack_y,
                   reservoir_names_in_node_order, worst_k=10, clamp_pred_to_fr=True,
                   exclude_reservoirs=(), monitor_reservoirs=(), overall_r2_mode="micro"):
    """
    Clean evaluation:
      - R2 on scaled space (before inverse_transform)
      - R2 on original space (after inverse_transform)
      - per-reservoir R2 and worst K list
      - worst-3 detail statistics in original space
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_persist = []  # (n_samples, n_nodes, n_days) in scaled-y space



    with torch.no_grad():
        for graph_batch, target_batch in test_loader:
            preds, targets, persists = [], [], []
            for graphs, tgt in zip(graph_batch, target_batch):
                graphs = [g.to(device) for g in graphs]

                reservoir_ids = None
                if bool(getattr(model, "use_reservoir_emb", False)):
                    reservoir_ids = torch.arange(int(graphs[0].x.size(0)), device=device, dtype=torch.long)

                pred_raw = model(graphs, reservoir_ids=reservoir_ids)
                pred_raw = _squeeze_pred_tensor(pred_raw)

                if torch.is_tensor(pred_raw) and pred_raw.dim() == 3:
                    if pred_raw.size(0) == 1:
                        pred_raw = pred_raw[0]
                    else:
                        raise RuntimeError(f"Unexpected pred shape in eval: {tuple(pred_raw.shape)}")

                pred_scaled = link_pred_to_scaled(pred_raw, inv_pack_y)
                if bool(clamp_pred_to_fr) and (inv_pack_y is not None) and (inv_pack_y.get("type") == "minmax"):
                    fr_min = float(inv_pack_y["fr_min"])
                    fr_max = float(inv_pack_y["fr_max"])
                    pred_scaled = torch.clamp(pred_scaled, min=fr_min, max=fr_max)

                preds.append(pred_scaled)
                H = int(pred_scaled.size(-1))
                persist_scaled = graphs[-1].x[:, 0].unsqueeze(-1).repeat(1, H)
                persists.append(persist_scaled)
                targets.append(tgt.to(device))
            preds = torch.stack(preds)
            targets = torch.stack(targets)
            persists = torch.stack(persists)
            all_persist.append(persists.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)  # (samples, nodes, 7)
    all_targets = np.concatenate(all_targets, axis=0)

    all_persist = np.concatenate(all_persist, axis=0)  # (samples, nodes, 7)


    n_samples, n_nodes, n_days = all_predictions.shape
    idx_to_reservoir = _build_idx_to_reservoir(encode_map)
    
    # consistency check on the scaled targets before any inverse transform
    debug_scaled_impossibility(
        targets_scaled=all_targets,   # scaled targets collected above, (S,N,T)
        scaler_data=scaler_data,
        reservoir_names_in_node_order=reservoir_names_in_node_order,
        k=15
    )

    # ensure we have a name for every node index (prevents missing keys downstream)
    for i in range(n_nodes):
        if i not in idx_to_reservoir:
            idx_to_reservoir[i] = f"Node_{i}"

    # SUBSET FILTER (manual exclude list)
    exclude_names = set(exclude_reservoirs or [])
    if exclude_names:
        not_found = sorted([n for n in exclude_names if (encode_map is None or n not in encode_map)])
        if not_found:
            print("[WARN] EXCLUDE_RESERVOIRS not in encode_map (ignored):", not_found[:20])

    keep_node_idx = [i for i in range(n_nodes) if idx_to_reservoir[i] not in exclude_names]
    if len(keep_node_idx) == 0:
        raise RuntimeError("All reservoirs are excluded. EXCLUDE_RESERVOIRS leaves empty set.")
    print(f"[SUBSET] scoring on {len(keep_node_idx)}/{n_nodes} reservoirs; excluded={len(exclude_names)}")
    if exclude_names:
        print("[SUBSET] excluded:", sorted(list(exclude_names)))

    # -------------------------
    # Scaled space
    # -------------------------
    print("\n" + "=" * 60)
    print("EVALUATION (SCALED SPACE, BEFORE INVERSE TRANSFORM)")
    print("=" * 60)
    print(f"Samples: {n_samples} | Reservoirs: {n_nodes} | Horizon(days): {n_days}")

    tgt_scaled_keep  = all_targets[:, keep_node_idx, :]
    pred_scaled_keep = all_predictions[:, keep_node_idx, :]

    overall_r2_scaled = r2_score(tgt_scaled_keep.reshape(-1), pred_scaled_keep.reshape(-1))
    daily_r2_scaled = []
    for d in range(n_days):
        daily_r2_scaled.append(r2_score(tgt_scaled_keep[:, :, d].reshape(-1), pred_scaled_keep[:, :, d].reshape(-1)))

    # per-reservoir R2 is computed over every node first and filtered afterwards, so that
    # the encode_map contract is left untouched
    r2_scaled_dict_all, worst_scaled_all = _per_reservoir_r2(all_predictions, all_targets, idx_to_reservoir)
    worst_scaled = [(k, v) for (k, v) in worst_scaled_all if k not in exclude_names]

    print(f"Overall R2 (scaled): {overall_r2_scaled:.4f}")
    print("Daily R2 (scaled): " + ", ".join([f"{x:.4f}" for x in daily_r2_scaled]))
    print("Worst reservoirs by R2 (scaled): " + ", ".join([f"{k}:{v:.3f}" for k, v in worst_scaled[:worst_k]]))

    # -------------------------
    # Original space
    # -------------------------
    print("\n" + "=" * 60)
    print("EVALUATION (ORIGINAL SPACE, AFTER INVERSE TRANSFORM)")
    print("=" * 60)

    pred_org, true_org = inverse_transform_predictions(all_predictions, all_targets, scaler_data, encode_map)

    true_org_keep = true_org[:, keep_node_idx, :]
    pred_org_keep = pred_org[:, keep_node_idx, :]

    # ---- baseline: persistence R2 in ORIGINAL space (after inverse transform) ----
    persist_org, true_org_for_persist = inverse_transform_predictions(
        predictions=all_persist,
        targets=all_targets,
        scaler_data=scaler_data,
        encode_map=encode_map,
    )

    # define keep tensors locally to avoid order-dependence
    true_org_keep_local = true_org[:, keep_node_idx, :]
    persist_org_keep_local = persist_org[:, keep_node_idx, :]

    # ---- BASELINE: persistence R2 in ORIGINAL space (daily d1..d7) ----
    persist_org, _true_org_unused = inverse_transform_predictions(
        all_persist, all_targets, scaler_data, encode_map
    )
    persist_org_keep = persist_org[:, keep_node_idx, :]

    monitor_names = list(monitor_reservoirs or [])
    keep_names = [idx_to_reservoir[i] for i in keep_node_idx]

    # overall daily R2 for persistence (original)
    persist_daily_r2_org = []
    for d in range(n_days):
        persist_daily_r2_org.append(
            r2_score(
                true_org_keep[:, :, d].reshape(-1),
                persist_org_keep[:, :, d].reshape(-1),
            )
        )
    print(
        "[BASELINE][persist][original][daily][overall] "
        + ", ".join([f"d{d+1}={v:.4f}" for d, v in enumerate(persist_daily_r2_org)])
    )

    # per-reservoir daily R2 for persistence (original) on monitor list
    for _name in monitor_names:
        if _name not in keep_names:
            print(f"[BASELINE][persist][original][daily] {_name}: not in keep set")
            continue
        _j = keep_names.index(_name)
        _vals = []
        for d in range(n_days):
            _vals.append(
                r2_score(
                    true_org_keep[:, _j, d].reshape(-1),
                    persist_org_keep[:, _j, d].reshape(-1),
                )
            )
        print(
            f"[BASELINE][persist][original][daily] {_name}: "
            + ", ".join([f"d{d+1}={v:.4f}" for d, v in enumerate(_vals)])
        )


    overall_persist_r2_org = r2_score(
        true_org_keep_local.reshape(-1),
        persist_org_keep_local.reshape(-1),
    )
    print(f"[BASELINE][persist][original] overall_r2={overall_persist_r2_org:.4f}")

    monitor_names = list(monitor_reservoirs or [])

    keep_names = [idx_to_reservoir[i] for i in keep_node_idx]
    for _name in monitor_names:
        if _name in keep_names:
            _j = keep_names.index(_name)
            _r2 = r2_score(
                true_org_keep_local[:, _j, :].reshape(-1),
                persist_org_keep_local[:, _j, :].reshape(-1),
            )
            print(f"[BASELINE][persist][original] {_name} r2={_r2:.4f}")

    overall_r2_org = r2_score(true_org_keep.reshape(-1), pred_org_keep.reshape(-1))
    daily_r2_org = []
    for d in range(n_days):
        daily_r2_org.append(r2_score(true_org_keep[:, :, d].reshape(-1), pred_org_keep[:, :, d].reshape(-1)))

    r2_org_dict_all, worst_org_all = _per_reservoir_r2(pred_org, true_org, idx_to_reservoir)
    r2_org_daily_dict_all = _per_reservoir_r2_daily(pred_org, true_org, idx_to_reservoir)

    # original-space metrics: overall and daily, equal weight across reservoirs

    # 1) drop excluded reservoirs first, so the macro average is taken over the kept ones
    r2_org_dict = {k: v for k, v in r2_org_dict_all.items() if k not in exclude_names}
    r2_org_daily_dict = {k: v for k, v in r2_org_daily_dict_all.items() if k not in exclude_names}
    worst_org = [(k, v) for (k, v) in worst_org_all if k not in exclude_names]

    # 2) keep the micro values, computed over the flattened pool
    overall_r2_micro_org = overall_r2_org
    daily_r2_micro_org = daily_r2_org

    # 3) macro: equal weight per reservoir
    _vals = np.array(list(r2_org_dict.values()), dtype=np.float64)
    overall_r2_org = float(np.nanmean(_vals)) if _vals.size else np.nan

    _daily_mat = np.array(list(r2_org_daily_dict.values()), dtype=np.float64)  # (n_keep, n_days)
    daily_r2_org = list(np.nanmean(_daily_mat, axis=0)) if _daily_mat.size else [np.nan] * n_days

    print(f"Overall R2 (original, macro): {overall_r2_org:.4f}")
    print(f"Overall R2 (original, micro): {overall_r2_micro_org:.4f}")
    print("Daily R2 (original, macro): " + ", ".join([f"{x:.4f}" for x in daily_r2_org]))
    print("Daily R2 (original, micro): " + ", ".join([f"{x:.4f}" for x in daily_r2_micro_org]))
    print("Worst reservoirs by R2 (original): " + ", ".join([f"{k}:{v:.3f}" for k, v in worst_org[:worst_k]]))

    # -------------------------
    # Summary
    # -------------------------

    neg_scaled = sum(1 for _, v in worst_scaled if v < 0)
    neg_org = sum(1 for _, v in worst_org if v < 0)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (SCALED vs ORIGINAL)")
    print("=" * 60)
    print(f"Overall R2 - scaled:   {overall_r2_scaled:.4f}")
    print(f"Overall R2 - original (macro): {overall_r2_org:.4f}")
    print(f"Overall R2 - original (micro): {overall_r2_micro_org:.4f}")
    print(f"Δ (micro original - scaled): {overall_r2_micro_org - overall_r2_scaled:.4f}")
    print(f"Negative-R2 reservoirs - scaled: {neg_scaled}, original: {neg_org}")

    # -------------------------
    # Worst-3 details (original)
    # -------------------------
    print("\n" + "=" * 60)
    print("DETAIL (WORST 3 RESERVOIRS, ORIGINAL SPACE)")
    print("=" * 60)

    for name, r2v in worst_org[:3]:
        node_idx = encode_map[name] if (encode_map and name in encode_map) else None
        if node_idx is None:
            # fallback: find by idx_to_reservoir reverse lookup
            inv_map = {v: k for k, v in idx_to_reservoir.items()}
            node_idx = inv_map.get(name, None)
        if node_idx is None:
            continue

        y_true = true_org[:, node_idx, :].reshape(-1)
        y_pred = pred_org[:, node_idx, :].reshape(-1)

        print(f"\n{name} | R2(original)={r2v:.4f}")
        print(f"  y_true: mean={y_true.mean():.3f}, std={y_true.std():.3f}, range=[{y_true.min():.3f}, {y_true.max():.3f}]")
        print(f"  y_pred: mean={y_pred.mean():.3f}, std={y_pred.std():.3f}, range=[{y_pred.min():.3f}, {y_pred.max():.3f}]")
        if (y_pred < 0).any():
            print("  FLAG: negative predictions exist (original space).")

    # which overall/daily pair is exported as the headline R2
    _mode = str(overall_r2_mode).lower().strip()
    if _mode not in ("micro", "macro"):
        raise ValueError(f"OVERALL_R2_MODE must be 'micro' or 'macro', got: {_mode}")

    # at this point overall_r2_org / daily_r2_org hold the macro values, and
    # overall_r2_micro_org / daily_r2_micro_org hold the micro ones
    if _mode == "micro":
        overall_r2_out = overall_r2_micro_org
        daily_r2_out = daily_r2_micro_org
    else:
        overall_r2_out = overall_r2_org
        daily_r2_out = daily_r2_org

    print(f"[REPORT] Overall R2 Score uses {_mode}.")
    return overall_r2_out, daily_r2_out, r2_org_dict, r2_org_daily_dict

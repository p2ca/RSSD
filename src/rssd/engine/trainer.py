"""Source training: one epoch of the objective, and in-training evaluation.

``run_epoch`` carries the complete training objective: the smooth-L1 prediction loss,
averaged equally over the batch and the seven lead days, the shared-basis regulariser, and
whichever whole-latent alignment term the variant uses (MMD, CORAL or domain-adversarial).
The prediction clamp and gradient clipping are keyword arguments. ``evaluate_model`` is the
in-training source evaluation; the target-side evaluation with adaptation lives in
:mod:`rssd.engine.evaluator`.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm

from rssd.common import _squeeze_pred_tensor, link_pred_to_scaled
from rssd.data.reservoirs import _build_idx_to_reservoir
from rssd.data.scalers import debug_scaled_impossibility, inverse_y_scaled_to_phys_torch
from rssd.metrics import _per_reservoir_r2, _per_reservoir_r2_daily, _per_reservoir_r2_log1p
from rssd.objectives.alignment import _coral_loss, _dann_loss, _mmd_rbf, _window_mean_latent
from rssd.utils import inverse_transform_predictions

__all__ = ["_pack_collated_batch", "run_epoch", "evaluate_model"]


def _pack_collated_batch(graph_batch, target_batch=None):
    """
    graph_batch: length=B, each item is a list(len=Tin) of graphs
    target_batch: optional length=B, each item is tgt tensor
    returns:
      graphs_packed, tgt_packed(optional), Bwin, base_nodes
    """
    graphs_list = list(graph_batch)
    Bwin = len(graphs_list)
    Tin = len(graphs_list[0])
    base_nodes = int(graphs_list[0][0].x.shape[0])

    graphs_packed = []
    for t in range(Tin):
        x_cat = torch.cat([graphs_list[b][t].x for b in range(Bwin)], dim=0)
        g_ref = graphs_list[0][t]
        g_new = g_ref.clone() if hasattr(g_ref, "clone") else copy.copy(g_ref)
        g_new.x = x_cat
        graphs_packed.append(g_new)

    tgt_packed = None
    if target_batch is not None:
        tgt_list = list(target_batch)
        tgt_packed = torch.cat([_squeeze_pred_tensor(tt) for tt in tgt_list], dim=0)

    return graphs_packed, tgt_packed, Bwin, base_nodes


def run_epoch(
    model,
    loader,
    criterion,
    inv_pack_y,
    y_transform,
    y_scale_t,
    optimizer=None,
    train=True,
    device="cpu",
    epoch: int = 0,
    log_every: int = 0,
    target_loader=None,
    align_method: str = "none",
    align_weight: float = 0.0,
    domain_discriminator=None,
    mmd_weight: float = 0.0,
    mmd_kernel_num: int = 5,
    mmd_kernel_mul: float = 2.0,
    mmd_normalize_latent: bool = True,
    mmd_max_samples_per_domain: int = 0,
    DARSD_WEIGHT: float = 2e-4,
    clamp_pred_to_fr: bool = True,
    grad_clip_norm: float = 0.0,
):
    model.train() if train else model.eval()
    align_method = str(align_method or "none").lower()
    use_align_train = bool(
        train and (align_method in ("mmd", "coral", "dann"))
        and (target_loader is not None) and (float(align_weight) > 0.0)
    )
    if domain_discriminator is not None:
        domain_discriminator.train() if train else domain_discriminator.eval()
    target_iter = iter(target_loader) if use_align_train else None
    total_loss = 0.0
    align_sum = 0.0
    align_cnt = 0
    n_steps = 0
    mae_sum = 0.0
    mae_cnt = 0

    global_step = 0
    eps = 1e-6
    
    grad_clip_norm = float(grad_clip_norm or 0.0)

    with torch.set_grad_enabled(train):
        for batch_idx, (graph_batch, target_batch) in enumerate(tqdm(loader, desc=f"{'Train' if train else 'Eval'} Epoch {epoch}")):
            # graph_batch: length=B, each item is a list(len=Tin) of graphs with g.x=(nodes,F)
            # target_batch: length=B, each item is tgt with shape (nodes,pred_len) (or squeezable)
            graphs_packed, tgt_packed, Bwin, base_nodes = _pack_collated_batch(graph_batch, target_batch)

            # wrap as a single-iteration list so we can reuse the existing code below with minimal edits
            packed_pairs = [(graphs_packed, tgt_packed, Bwin, base_nodes)]

            for inbatch_idx, (graphs, tgt, _Bwin, _base_nodes) in enumerate(packed_pairs):
                step_idx = global_step
                global_step += 1

                graphs = [g.to(device) for g in graphs]
                tgt = tgt.to(device)

                # --- source reservoir_ids for reservoir embedding (critical for packed windows) ---
                reservoir_ids = None
                _need_reservoir_ids = bool(
                    getattr(model, "use_reservoir_emb", False)
                    or getattr(model, "use_meta_only_static", False)
                )
                if _need_reservoir_ids:
                    if ("_Bwin" in locals()) and ("_base_nodes" in locals()) and (_Bwin is not None) and (_base_nodes is not None):
                        reservoir_ids = torch.arange(int(_base_nodes), device=device, dtype=torch.long).repeat(int(_Bwin))
                    else:
                        reservoir_ids = torch.arange(int(graphs[0].x.size(0)), device=device, dtype=torch.long)

                # --- optional target batch for domain alignment (mmd/coral/dann) ---
                tgt_graphs_mmd = None
                tgt_reservoir_ids = None
                if use_align_train:
                    try:
                        target_graph_batch, _ = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_loader)
                        target_graph_batch, _ = next(target_iter)

                    tgt_graphs_mmd, _, tgt_Bwin, tgt_base_nodes = _pack_collated_batch(target_graph_batch, None)
                    tgt_graphs_mmd = [g.to(device) for g in tgt_graphs_mmd]

                    if bool(getattr(model, "use_reservoir_emb", False) or getattr(model, "use_meta_only_static", False)):
                        tgt_reservoir_ids = torch.arange(
                            int(tgt_base_nodes), device=device, dtype=torch.long
                        ).repeat(int(tgt_Bwin))
                
                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                # forward
                h_latent = None
                need_latent = bool(use_align_train and (tgt_graphs_mmd is not None))

                if need_latent:
                    y_hat, h_latent = model(
                        graphs,
                        return_latent=True,
                        reservoir_ids=reservoir_ids,
                    )
                else:
                    y_hat = model(graphs, reservoir_ids=reservoir_ids)

                # squeeze to 2D (nodes, pred_len)
                y_hat_raw = _squeeze_pred_tensor(y_hat)
                tgt2 = _squeeze_pred_tensor(tgt)

                if y_hat_raw.dim() == 3:
                    if y_hat_raw.size(0) == 1:
                        y_hat_raw = y_hat_raw[0]
                    else:
                        raise RuntimeError(f"Unexpected y_hat shape: {tuple(y_hat_raw.shape)}")
                if tgt2.dim() == 3:
                    if tgt2.size(0) == 1:
                        tgt2 = tgt2[0]
                    else:
                        raise RuntimeError(f"Unexpected tgt shape: {tuple(tgt2.shape)}")

                if y_hat_raw.shape != tgt2.shape:
                    raise RuntimeError(f"Shape mismatch: y_hat={tuple(y_hat_raw.shape)} tgt={tuple(tgt2.shape)}")

                # -------- main loss pipeline --------
                # expand the per-node inverse pack and scale to match the y_hat_raw nodes
                inv_pack_y_step = inv_pack_y
                y_scale_step = y_scale_t

                _baseN = int(y_scale_t.numel())
                _curN  = int(y_hat_raw.size(0))
                if _curN != _baseN:
                    if (_curN % _baseN) != 0:
                        raise RuntimeError(f"[pack] nodes mismatch: y_hat_raw has {_curN}, but y_scale_t has {_baseN}")
                    _rep = _curN // _baseN

                    # repeat y_scale
                    y_scale_step = y_scale_t.repeat(_rep)

                    # repeat inv_pack tensors (minmax / standard)
                    inv_pack_y_step = dict(inv_pack_y)  # shallow copy
                    if inv_pack_y["type"] == "minmax":
                        inv_pack_y_step["dmin"] = inv_pack_y["dmin"].repeat(_rep)
                        inv_pack_y_step["dmax"] = inv_pack_y["dmax"].repeat(_rep)
                        # fr_min/fr_max are scalars, keep as-is
                    elif inv_pack_y["type"] == "standard":
                        inv_pack_y_step["mu"]  = inv_pack_y["mu"].repeat(_rep)
                        inv_pack_y_step["std"] = inv_pack_y["std"].repeat(_rep)
                    else:
                        raise RuntimeError(f"unknown inv_pack_y type: {inv_pack_y.get('type')}")
                
                y_hat_scaled = link_pred_to_scaled(y_hat_raw, inv_pack_y_step)

                if bool(clamp_pred_to_fr) and (inv_pack_y_step is not None) and (inv_pack_y_step.get("type") == "minmax"):
                    fr_min = float(inv_pack_y_step["fr_min"])
                    fr_max = float(inv_pack_y_step["fr_max"])
                    y_hat_scaled = torch.clamp(y_hat_scaled, min=fr_min, max=fr_max)

                y_hat_phys = inverse_y_scaled_to_phys_torch(y_hat_scaled, inv_pack_y_step, y_transform)
                tgt_phys   = inverse_y_scaled_to_phys_torch(tgt2,         inv_pack_y_step, y_transform)

                scale_floor = torch.quantile(y_scale_step, 0.10)
                den = (torch.clamp(y_scale_step, min=scale_floor)[:, None] + eps)

                y_hat_norm = y_hat_phys / den
                tgt_norm   = tgt_phys   / den

                per_elem = criterion(y_hat_norm, tgt_norm)
                if per_elem.dim() != 2:
                    raise RuntimeError(f"Expected per_elem 2D (nodes,pred_len), got {tuple(per_elem.shape)}")

                # every lead day contributes equally
                pred_loss = per_elem.mean()

                # -------- domain-alignment loss (optional; training only, source->target) --------
                # mmd (exp3) / coral (exp8) / dann (exp9) share the same domain-agnostic,
                # window-mean latent extraction; only the loss formula differs.
                align_loss = None
                if use_align_train:
                    if tgt_graphs_mmd is None:
                        raise RuntimeError("alignment enabled but tgt_graphs_mmd is None. Check target loader logic.")

                    h_source_mmd = model.encode_latent(
                        graphs,
                        reservoir_ids=None,
                        use_domain_cond=False,
                        apply_darsd=False,
                    )
                    h_target_mmd = model.encode_latent(
                        tgt_graphs_mmd,
                        reservoir_ids=None,
                        use_domain_cond=False,
                        apply_darsd=False,
                    )

                    h_source_mmd = _window_mean_latent(h_source_mmd, _Bwin, _base_nodes)
                    h_target_mmd = _window_mean_latent(h_target_mmd, tgt_Bwin, tgt_base_nodes)

                    if align_method == "mmd":
                        align_loss = _mmd_rbf(
                            h_source_mmd,
                            h_target_mmd,
                            kernel_mul=float(mmd_kernel_mul),
                            kernel_num=int(mmd_kernel_num),
                            normalize_latent=bool(mmd_normalize_latent),
                            max_samples_per_domain=int(mmd_max_samples_per_domain),
                        )
                    elif align_method == "coral":
                        align_loss = _coral_loss(
                            h_source_mmd,
                            h_target_mmd,
                            normalize_latent=bool(mmd_normalize_latent),
                        )
                    elif align_method == "dann":
                        align_loss = _dann_loss(
                            domain_discriminator,
                            h_source_mmd,
                            h_target_mmd,
                            grl_lambda=float(align_weight),
                            normalize_latent=bool(mmd_normalize_latent),
                        )

                # -------- total loss --------
                loss = pred_loss
                # shared-basis regularizer (training only)
                if train and getattr(model, "use_darsd", False):
                    loss = loss + DARSD_WEIGHT * model.darsd_regularizer(entropy_weight=0.02)
                if align_loss is not None:
                    if align_method == "dann":
                        # GRL already scales the encoder gradient by align_weight (lambda);
                        # add the domain-classification loss with unit coefficient.
                        loss = loss + align_loss
                    else:
                        loss = loss + float(align_weight) * align_loss
                if train and optimizer is not None:
                    loss.backward()
                    if grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

                # -------- metrics --------
                abs_err = (y_hat_norm.detach() - tgt_norm.detach()).abs()
                mae_sum += float(abs_err.sum().item())
                mae_cnt += int(abs_err.numel())

                total_loss += float(loss.detach().item())
                if align_loss is not None:
                    align_sum += float(align_loss.detach().item())
                    align_cnt += 1
                n_steps += 1

                if log_every and train and (step_idx % log_every == 0):
                    pass

    avg_loss = total_loss / max(1, n_steps)
    normMAE_mean = mae_sum / max(1, mae_cnt)

    if use_align_train:
        avg_align = align_sum / max(1, align_cnt)
        print(f"[ALIGN:{align_method}][{'train' if train else 'eval'}][E{epoch:03d}] avg_align={avg_align:.6f} weight={float(align_weight):.6f}")

    return avg_loss, normMAE_mean


def evaluate_model(model, test_loader, encode_map, scaler_data, device, inv_pack_y,
                   reservoir_names_in_node_order, worst_k=10, clamp_pred_to_fr=True):
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

    with torch.no_grad():
        for graph_batch, target_batch in test_loader:
            preds, targets = [], []
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
                targets.append(tgt.to(device))
            preds = torch.stack(preds)
            targets = torch.stack(targets)
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)  # (samples, nodes, 7)
    all_targets = np.concatenate(all_targets, axis=0)

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

    # -------------------------
    # Scaled space
    # -------------------------
    print("\n" + "=" * 60)
    print("EVALUATION (SCALED SPACE, BEFORE INVERSE TRANSFORM)")
    print("=" * 60)
    print(f"Samples: {n_samples} | Reservoirs: {n_nodes} | Horizon(days): {n_days}")

    overall_r2_scaled = r2_score(all_targets.reshape(-1), all_predictions.reshape(-1))
    daily_r2_scaled = []
    for d in range(n_days):
        daily_r2_scaled.append(r2_score(all_targets[:, :, d].reshape(-1), all_predictions[:, :, d].reshape(-1)))

    _, worst_scaled = _per_reservoir_r2(all_predictions, all_targets, idx_to_reservoir)

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
    overall_r2_org = r2_score(true_org.reshape(-1), pred_org.reshape(-1))
    daily_r2_org = []
    for d in range(n_days):
        daily_r2_org.append(r2_score(true_org[:, :, d].reshape(-1), pred_org[:, :, d].reshape(-1)))

    r2_org_dict, worst_org = _per_reservoir_r2(pred_org, true_org, idx_to_reservoir)
    r2_org_daily_dict = _per_reservoir_r2_daily(pred_org, true_org, idx_to_reservoir)

    print(f"Overall R2 (original): {overall_r2_org:.4f}")
    print("Daily R2 (original): " + ", ".join([f"{x:.4f}" for x in daily_r2_org]))
    print("Worst reservoirs by R2 (original): " + ", ".join([f"{k}:{v:.3f}" for k, v in worst_org[:worst_k]]))

    # -------------------------
    # Summary
    # -------------------------

    # -------------------------
    # Log(y+1)-space R²
    # -------------------------
    r2_log_dict, worst_log = _per_reservoir_r2_log1p(pred_org, true_org, idx_to_reservoir)
    _log_vals = np.array(list(r2_log_dict.values()), dtype=np.float64)
    overall_r2_log = float(np.nanmean(_log_vals)) if _log_vals.size else np.nan
    print("\n" + "=" * 60)
    print("LOG(y+1)-SPACE R\u00b2  (negatives clipped to 0 before log1p)")
    print("=" * 60)
    print(f"Overall R\u00b2 log(y+1) (macro): {overall_r2_log:.4f}")
    for _rn in sorted(r2_log_dict):
        print(f"  {_rn}: {r2_log_dict[_rn]:.4f}")
    print("Worst by log-R\u00b2: " + ", ".join([f"{k}:{v:.3f}" for k, v in worst_log[:worst_k]]))

    neg_scaled = sum(1 for _, v in worst_scaled if v < 0)
    neg_org = sum(1 for _, v in worst_org if v < 0)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (SCALED vs ORIGINAL)")
    print("=" * 60)
    print(f"Overall R2 - scaled:   {overall_r2_scaled:.4f}")
    print(f"Overall R2 - original: {overall_r2_org:.4f}")
    print(f"Δ (original - scaled): {overall_r2_org - overall_r2_scaled:.4f}")
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

    return overall_r2_org, daily_r2_org, r2_org_dict, r2_org_daily_dict, r2_log_dict

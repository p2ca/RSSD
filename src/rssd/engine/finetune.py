"""Target-history fine-tuning: the adaptation step every transfer scenario runs.

Every adaptation setting is an explicit keyword argument, defaulting to the frozen
protocol values: full-parameter adaptation, a minimum improvement of 1e-5, and the target
batches cached on the device. Adaptation runs on the complete support block and is
stopped on the target's validation block.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rssd.common import _squeeze_pred_tensor, link_pred_to_scaled, set_seed
from rssd.data.scalers import inverse_y_scaled_to_phys_torch

__all__ = ["_finetune_configure_trainable_params", "run_epoch", "finetune_on_target"]


def _finetune_configure_trainable_params(model, mode="full"):
    """Return list of (name, param) tuples to optimize during finetune."""
    if mode == "full":
        return [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    elif mode == "emb_head":
        return [(n, p) for n, p in model.named_parameters()
                if any(k in n for k in ("reservoir_emb", "res_static", "meta_static",
                                         "fc1", "fc2", "metadata"))]
    elif mode == "emb":
        return [(n, p) for n, p in model.named_parameters()
                if any(k in n for k in ("reservoir_emb", "res_static", "meta_static"))]
    elif mode == "head":
        return [(n, p) for n, p in model.named_parameters()
                if any(k in n for k in ("fc1", "fc2"))]
    else:
        return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


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
):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_steps = 0
    mae_sum = 0.0
    mae_cnt = 0

    global_step = 0
    eps = 1e-6

    with torch.set_grad_enabled(train):
        for batch_idx, (graph_batch, target_batch) in enumerate(
            tqdm(loader, desc=f"{'Train' if train else 'Eval'} Epoch {epoch}")
        ):
            for inbatch_idx, (graphs, tgt) in enumerate(zip(graph_batch, target_batch)):
                step_idx = global_step
                global_step += 1

                graphs = [g.to(device) for g in graphs]
                tgt = tgt.to(device)

                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                # -------------------
                # forward
                # -------------------
                y_hat = model(graphs)

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
                y_hat_scaled = link_pred_to_scaled(y_hat_raw, inv_pack_y)

                y_hat_phys = inverse_y_scaled_to_phys_torch(y_hat_scaled, inv_pack_y, y_transform)
                tgt_phys   = inverse_y_scaled_to_phys_torch(tgt2,         inv_pack_y, y_transform)

                scale_floor = torch.quantile(y_scale_t, 0.10)
                den = (torch.clamp(y_scale_t, min=scale_floor)[:, None] + eps)

                y_hat_norm = y_hat_phys / den
                tgt_norm   = tgt_phys   / den

                pred_loss = criterion(y_hat_norm, tgt_norm)

                # the shared-basis regularizer is a source-training term only
                loss = pred_loss

                if train and optimizer is not None:
                    loss.backward()
                    optimizer.step()

                # -------- metrics --------
                abs_err = (y_hat_norm.detach() - tgt_norm.detach()).abs()
                mae_sum += float(abs_err.sum().item())
                mae_cnt += int(abs_err.numel())

                total_loss += float(loss.detach().item())
                n_steps += 1

                if log_every and train and (step_idx % log_every == 0):
                    pass

    avg_loss = total_loss / max(1, n_steps)
    normMAE_mean = mae_sum / max(1, mae_cnt)
    return avg_loss, normMAE_mean


def finetune_on_target(
    model,
    train_dataset_full,
    val_dataset,
    collate_fn,
    device,
    inv_pack_y,
    max_epochs=10,
    lr=3e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    patience=2,
    seed=20260312,
    mode="full",
    batch_size=None,
    eval_batch_size=128,
    num_workers=0,
    pin_memory=False,
    cache_to_device=True,
    train_max_batches=None,
    val_max_batches=None,
    clamp_pred_to_fr=True,
    min_delta=1e-5,
):
    """
    Uses the SAME prediction contract as evaluation: raw -> scaled via link_pred_to_scaled, then optional clamp.
    """
    from torch.utils.data import DataLoader

    set_seed(int(seed))
    model = model.to(device)

    ft_train_ds = train_dataset_full
    ft_val_ds = val_dataset
    print(f"[FINETUNE] adaptation={len(ft_train_ds)} validation={len(ft_val_ds)}")

    bs = int(batch_size or eval_batch_size)
    num_workers = int(num_workers)
    pin_memory = bool(pin_memory)

    ft_train_loader = DataLoader(
        ft_train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    ft_val_loader = DataLoader(
        ft_val_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )

    def _cache_loader(loader, max_batches=None):
        cached = []
        for bi, (graph_batch, target_batch) in enumerate(loader):
            if (max_batches is not None) and (bi >= int(max_batches)):
                break
            # move whole batch to device ONCE
            graph_batch_dev = []
            for graphs in graph_batch:
                graph_batch_dev.append([g.to(device) for g in graphs])
            target_batch_dev = [t.to(device) for t in target_batch]
            cached.append((graph_batch_dev, target_batch_dev))
        return cached

    cached_train = None
    cached_val = None
    if bool(cache_to_device):
        cached_train = _cache_loader(ft_train_loader, train_max_batches)
        cached_val   = _cache_loader(ft_val_loader,   val_max_batches)
        print(f"[FINETUNE] cached batches: train={len(cached_train)} val={len(cached_val)}")

    trainables = _finetune_configure_trainable_params(model, str(mode))
    if len(trainables) == 0:
        raise RuntimeError("[FINETUNE] No trainable parameters selected. Check FINETUNE_MODE.")

    opt = torch.optim.AdamW(
        [p for _, p in trainables],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    best_val = float("inf")
    best_sd = None
    bad = 0
    
    def _move_batch_to_device(graph_batch, target_batch):
        """Move one batch to device (used only when NOT cached)."""
        graph_batch_dev = []
        for graphs in graph_batch:
            graph_batch_dev.append([g.to(device) for g in graphs])
        target_batch_dev = [t.to(device) for t in target_batch]
        return graph_batch_dev, target_batch_dev

    def _batch_loss(graph_batch_dev, target_batch_dev):
        """
        Compute loss assuming graph_batch_dev/target_batch_dev are already on device.
        (Quality-identical; avoids repeated .to(device) when cached.)
        """
        losses = []
        for graphs, tgt in zip(graph_batch_dev, target_batch_dev):
            y_hat = model(graphs)
            y_hat = _squeeze_pred_tensor(y_hat)  # (nodes, pred_len)
            
            pred_scaled = link_pred_to_scaled(y_hat, inv_pack_y)
            if bool(clamp_pred_to_fr) and (inv_pack_y is not None) and (inv_pack_y.get("type") == "minmax"):
                fr_min = float(inv_pack_y["fr_min"])
                fr_max = float(inv_pack_y["fr_max"])
                pred_scaled = torch.clamp(pred_scaled, min=fr_min, max=fr_max)

            losses.append(torch.mean((pred_scaled - tgt) ** 2))
        return torch.stack(losses).mean()

    for epoch in range(int(max_epochs)):
        model.train()
        tr_losses = []
        train_cached = (cached_train is not None)
        train_iter = cached_train if train_cached else ft_train_loader

        for graph_batch, target_batch in train_iter:
            opt.zero_grad(set_to_none=True)

            if not train_cached:
                graph_batch, target_batch = _move_batch_to_device(graph_batch, target_batch)

            loss = _batch_loss(graph_batch, target_batch)
            loss.backward()

            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_([p for _, p in trainables], max_norm=float(grad_clip))

            opt.step()
            tr_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        va_losses = []
        val_cached = (cached_val is not None)
        val_iter = cached_val if val_cached else ft_val_loader

        with torch.no_grad():
            for graph_batch, target_batch in val_iter:
                if not val_cached:
                    graph_batch, target_batch = _move_batch_to_device(graph_batch, target_batch)
                loss = _batch_loss(graph_batch, target_batch)
                va_losses.append(float(loss.detach().cpu().item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")

        print(f"[FINETUNE] epoch={epoch+1:02d}/{int(max_epochs)}  train_mse={tr:.6f}  val_mse={va:.6f}")

        min_delta = float(min_delta)
        if va < best_val - min_delta:
            best_val = va
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                print(f"[FINETUNE] early stop (patience={patience})")
                break

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=False)
        print(f"[FINETUNE] restored best val_mse={best_val:.6f}")

    model.eval()
    return model

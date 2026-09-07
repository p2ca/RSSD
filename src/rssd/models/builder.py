"""Constructing a model, attaching its static-attribute buffers and loading source weights.

Training builds a model from an experiment configuration; evaluation builds it from the
checkpoint's own configuration (:mod:`rssd.models.checkpoint`) and then has to solve a
second problem: the target reservoirs are usually a different node set from the source
reservoirs the weights were trained on. The policy is:

* node-tied tensors (``reservoir_emb.weight``, ``res_static``) are never preloaded across
  a genuinely different node set;
* the same node set in a different order is remapped by reservoir name;
* an unseen target node set gets its embedding initialised either from the source mean or,
  when static attributes are available, from a ridge map fitted on the source reservoirs
  (attributes -> embedding), blended toward the source mean and clamped into the source
  norm band.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rssd.data.static_attrs import STATIC_EPS, build_static_matrix_raw, robust_norm_stats
from rssd.models.lstm import Seq2SeqLSTM

__all__ = [
    "resolve_device", "build_model", "build_model_from_checkpoint_config",
    "attach_static_attributes", "compare_node_sets", "load_source_weights",
    "initialize_target_embedding", "build_domain_discriminator", "build_optimizer",
]

# Backbone-specific weights that must survive the filtered load, or the checkpoint-locked
# parameters disagree with the bundle and every downstream number would be meaningless.
CORE_KEYS = {
    "transformer_seq2seq": [
        "input_encoder.0.weight",
        "encoder.layers.0.self_attn.in_proj_weight",
        "decoder.layers.0.self_attn.in_proj_weight",
        "decoder.layers.0.multihead_attn.in_proj_weight",
        "pos_src.weight", "pos_tgt.weight", "fc1.weight", "fc2.weight",
    ],
    "lstm": [
        "input_encoder.0.weight",
        "encoder.weight_ih_l0", "encoder.weight_hh_l0",
        "fc1.weight", "fc2_direct.weight",
    ],
}

NODE_TIED_KEYS = {"res_static", "reservoir_emb.weight"}


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(*, input_dim, hidden_dim, num_layers, pred_len, dropout, latent_mode,
                use_latent_proj, num_reservoirs=0,
                use_reservoir_emb=False, reservoir_emb_dim=0, emb_dropout_p=0.0,
                use_res_static=False, res_static_dim=0,
                use_meta_only_static=False, meta_only_static_dim=0,
                use_darsd=False, lcib_k=0,
                backbone="lstm", n_heads=8, tf_layers=2, tf_ff_mult=4, tin=30,
                device=None):
    """Instantiate ``Seq2SeqLSTM`` with the project's argument contract."""
    device = resolve_device(device)
    model = Seq2SeqLSTM(
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        output_dim=1,
        num_layers=int(num_layers),
        pred_len=int(pred_len),
        dropout=float(dropout),

        use_reservoir_emb=bool(use_reservoir_emb),
        num_reservoirs=int(num_reservoirs),
        reservoir_emb_dim=int(reservoir_emb_dim),
        emb_dropout_p=float(emb_dropout_p),

        use_res_static=bool(use_res_static),
        res_static_dim=int(res_static_dim),

        use_meta_only_static=bool(use_meta_only_static),
        meta_only_static_dim=int(meta_only_static_dim),

        latent_mode=str(latent_mode),
        use_latent_proj=bool(use_latent_proj),


        use_darsd=bool(use_darsd),
        lcib_k=int(lcib_k),

        backbone=str(backbone),
        n_heads=int(n_heads),
        tf_layers=int(tf_layers),
        tf_ff_mult=int(tf_ff_mult),
        tin=int(tin),
    ).to(device)
    return model


def build_model_from_checkpoint_config(cfg, *, input_dim, pred_len, num_nodes, device=None):
    """Build the model the bundle describes, sized for the current evaluation node set."""
    return build_model(
        input_dim=input_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        pred_len=pred_len, dropout=cfg.dropout, latent_mode=cfg.latent_mode,
        use_latent_proj=cfg.use_latent_proj,
        num_reservoirs=num_nodes, use_reservoir_emb=cfg.use_reservoir_emb,
        reservoir_emb_dim=cfg.reservoir_emb_dim, emb_dropout_p=cfg.emb_dropout_p,
        use_res_static=cfg.use_res_static, res_static_dim=cfg.res_static_dim,
        use_meta_only_static=cfg.use_meta_only_static,
        meta_only_static_dim=cfg.meta_only_static_dim,
        use_darsd=cfg.use_darsd, lcib_k=cfg.lcib_k,
        backbone=cfg.backbone, n_heads=cfg.n_heads, tf_layers=cfg.tf_layers,
        tf_ff_mult=cfg.tf_ff_mult, tin=cfg.tin, device=device,
    )


def attach_static_attributes(model, static_np, *, use_res_static, use_meta_only_static,
                             device=None):
    """Install the normalised attribute matrix into whichever buffer the variant uses."""
    if static_np is None or not (use_res_static or use_meta_only_static):
        return None
    device = resolve_device(device)
    tensor = torch.tensor(np.asarray(static_np, dtype=np.float32), dtype=torch.float32)
    if use_res_static:
        model.set_res_static(tensor.to(device))
        return "res_static"
    model.set_meta_static(tensor.to(device))
    return "meta_static"


def compare_node_sets(eval_names, train_names):
    """How the evaluation node set relates to the one the checkpoint was trained on."""
    eval_names, train_names = list(eval_names), list(train_names)
    same_order = eval_names == train_names
    same_nodes = len(eval_names) == len(train_names) and set(eval_names) == set(train_names)
    return {"same_order": same_order, "same_nodes": same_nodes,
            "eval_names": eval_names, "train_names": train_names}


def load_source_weights(model, state_dict, *, node_sets, backbone="lstm"):
    """Load source weights, skipping node-tied tensors on a true cross-dataset evaluation."""
    model_sd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state_dict.items():
        if (not node_sets["same_nodes"]) and (k in NODE_TIED_KEYS):
            skipped.append(k)
            continue
        if (k in model_sd) and (tuple(model_sd[k].shape) == tuple(v.shape)):
            filtered[k] = v
        else:
            skipped.append(k)

    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)

    core_keys = CORE_KEYS.get(str(backbone), CORE_KEYS["lstm"])
    core_missing = [k for k in core_keys if k not in filtered]
    if core_missing:
        raise RuntimeError(
            "[CKPT][FATAL] core weights NOT loaded (shape mismatch / wrong ckpt-locked params). "
            f"missing_core={core_missing}. Check hidden_dim/num_layers/use_darsd/"
            "lcib_k/res_static_dim/reservoir_emb_dim are ckpt-locked correctly.")

    return {"loaded": len(filtered), "skipped": skipped,
            "missing_keys": list(missing_keys), "unexpected_keys": list(unexpected_keys)}


def _fit_attribute_to_embedding_ridge(Xs_n, src_emb_np, ridge_lambda):
    """Ridge map from normalised source attributes to source embeddings (intercept free)."""
    n_src = Xs_n.shape[0]
    mu_y = src_emb_np.mean(axis=0, keepdims=True)
    Y = (src_emb_np - mu_y).astype(np.float32)
    X_aug = np.concatenate([Xs_n, np.ones((n_src, 1), dtype=np.float32)], axis=1)
    reg = float(ridge_lambda) * np.eye(X_aug.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    W = np.linalg.solve((X_aug.T @ X_aug) + reg, (X_aug.T @ Y)).astype(np.float32)
    return W, mu_y


def initialize_target_embedding(model, src_emb, *, node_sets, target_names, source_names,
                                feature_names, use_static, use_meta_emb_init=True,
                                ridge_lambda=1e-3, blend_alpha=0.5, norm_clip=3.0,
                                fatal_p95=0.0, align_dir=None, latlon_csv=None):
    """Give the evaluation node set an embedding consistent with the source checkpoint.

    Returns the method actually used: ``keep_ckpt_exact``, ``remap_from_ckpt_order``,
    ``meta2emb_ridge`` or ``mean``.
    """
    if src_emb is None or not getattr(model, "use_reservoir_emb", False):
        return "none"

    weight = model.reservoir_emb.weight
    n_tgt = int(weight.shape[0])

    if node_sets["same_nodes"]:
        src_name_to_idx = {name: i for i, name in enumerate(node_sets["train_names"])}
        remap_idx = torch.tensor([src_name_to_idx[name] for name in target_names], dtype=torch.long)
        weight.data.copy_(src_emb[remap_idx].to(weight.device, dtype=weight.dtype))
        return "keep_ckpt_exact" if node_sets["same_order"] else "remap_from_ckpt_order"

    if not (use_meta_emb_init and use_static):
        weight.data.copy_(src_emb.mean(dim=0, keepdim=True).repeat(n_tgt, 1).to(
            weight.device, dtype=weight.dtype))
        return "mean"

    try:
        Xs = build_static_matrix_raw(source_names, feature_names, align_dir, latlon_csv)
        med_s, iqr_s = robust_norm_stats(Xs)
        Xs_n = ((Xs - med_s) / (iqr_s + STATIC_EPS)).astype(np.float32)

        src_emb_np = src_emb.detach().cpu().numpy().astype(np.float32)
        if Xs_n.shape[0] != src_emb_np.shape[0]:
            raise RuntimeError(f"[meta2emb] source num_nodes mismatch: attributes={Xs_n.shape[0]} "
                               f"vs ckpt_emb={src_emb_np.shape[0]}")
        W, mu_y = _fit_attribute_to_embedding_ridge(Xs_n, src_emb_np, ridge_lambda)

        Xt = build_static_matrix_raw(target_names, feature_names, align_dir, latlon_csv)
        Xt_n = ((Xt - med_s) / (iqr_s + STATIC_EPS)).astype(np.float32)

        xt_p95 = float(np.percentile(np.abs(Xt_n), 95))
        if float(fatal_p95) > 0.0 and xt_p95 > float(fatal_p95):
            raise RuntimeError("[meta2emb][FATAL] target metadata is too far outside the source "
                               f"normalization range (abs-p95={xt_p95:.3f} > {float(fatal_p95):.3f}).")
        if float(norm_clip) > 0.0:
            Xt_n = np.clip(Xt_n, -float(norm_clip), float(norm_clip))

        Xt_aug = np.concatenate([Xt_n, np.ones((n_tgt, 1), dtype=np.float32)], axis=1)
        Yt = (Xt_aug @ W + mu_y).astype(np.float32)

        alpha = float(blend_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"blend_alpha must be in [0,1], got {alpha}")
        mean_emb = src_emb_np.mean(axis=0, keepdims=True)
        Yt = (1.0 - alpha) * mean_emb.repeat(n_tgt, axis=0) + alpha * Yt

        # keep the initialised rows inside the source embedding's norm band
        src_norm = np.linalg.norm(src_emb_np, axis=1)
        norm_lo, norm_hi = float(np.quantile(src_norm, 0.05)), float(np.quantile(src_norm, 0.95))
        tgt_norm = np.linalg.norm(Yt, axis=1, keepdims=True)
        Yt = Yt * (np.clip(tgt_norm, norm_lo, norm_hi) / np.clip(tgt_norm, 1e-6, None))

        weight.data.copy_(torch.tensor(Yt, dtype=weight.dtype, device=weight.device))
        return "meta2emb_ridge"
    except Exception as exc:  # a silent fallback here would corrupt every reported number
        raise RuntimeError("[CKPT][FATAL] attribute-to-embedding init failed during true "
                           f"cross-dataset eval. error = {exc}") from exc


def build_domain_discriminator(hidden_dim, disc_hidden, device=None, dropout=0.2):
    """Gradient-reversal discriminator used only by the domain-adversarial baseline."""
    device = resolve_device(device)
    return nn.Sequential(
        nn.Linear(int(hidden_dim), int(disc_hidden)),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(int(disc_hidden), 1),
    ).to(device)


def build_optimizer(model, *, lr, weight_decay, discriminator=None, disc_lr=None,
                    scheduler_factor=0.5, scheduler_patience=5, scheduler_min_lr=1e-5):
    """AdamW plus the plateau schedule used by every experiment."""
    param_groups = [{"params": model.parameters()}]
    if discriminator is not None:
        param_groups.append({"params": discriminator.parameters(), "lr": float(disc_lr)})
    optimizer = torch.optim.AdamW(param_groups, lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=scheduler_min_lr)
    return optimizer, scheduler

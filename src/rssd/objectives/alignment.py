"""Whole-latent domain-alignment objectives (MMD, CORAL, DANN) for the alignment baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["_GradReverse", "_grad_reverse", "_subsample_latent_for_mmd", "_window_mean_latent", "_estimate_mmd_bandwidth_from_sqdist", "_multi_rbf_from_sqdist", "_mmd_rbf", "_coral_loss", "_dann_loss"]

class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (DANN). Identity forward; grad * (-lambda) backward."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def _grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


def _subsample_latent_for_mmd(z, max_samples_per_domain=0):
    if z is None:
        return z
    max_samples_per_domain = int(max_samples_per_domain or 0)
    if max_samples_per_domain <= 0:
        return z
    if z.size(0) <= max_samples_per_domain:
        return z

    idx = torch.randperm(z.size(0), device=z.device)[:max_samples_per_domain]
    return z.index_select(0, idx)


def _window_mean_latent(h_latent: torch.Tensor, Bwin: int, base_nodes: int) -> torch.Tensor:
    if h_latent.dim() != 2:
        raise RuntimeError(f"_window_mean_latent expects 2D latent, got {tuple(h_latent.shape)}")
    expected = int(Bwin) * int(base_nodes)
    if int(h_latent.size(0)) != expected:
        raise RuntimeError(
            f"_window_mean_latent mismatch: latent rows={int(h_latent.size(0))}, "
            f"expected Bwin*base_nodes={expected}"
        )
    return h_latent.reshape(int(Bwin), int(base_nodes), int(h_latent.size(1))).mean(dim=1)


def _estimate_mmd_bandwidth_from_sqdist(sq_xx, sq_yy, sq_xy):
    vals = []

    if sq_xx.size(0) > 1:
        denom_xx = sq_xx.numel() - sq_xx.size(0)
        if denom_xx > 0:
            vals.append((sq_xx.sum() - sq_xx.diagonal().sum()) / float(denom_xx))

    if sq_yy.size(0) > 1:
        denom_yy = sq_yy.numel() - sq_yy.size(0)
        if denom_yy > 0:
            vals.append((sq_yy.sum() - sq_yy.diagonal().sum()) / float(denom_yy))

    if sq_xy.numel() > 0:
        vals.append(sq_xy.mean())

    if len(vals) == 0:
        return torch.tensor(1.0, device=sq_xy.device, dtype=sq_xy.dtype)

    bandwidth = torch.stack(vals).mean()
    return torch.clamp(bandwidth, min=1e-6)


def _multi_rbf_from_sqdist(sqdist, bandwidth, kernel_mul=2.0, kernel_num=5):
    bandwidth = torch.clamp(bandwidth, min=1e-6)
    base = bandwidth / (kernel_mul ** (kernel_num // 2))

    out = torch.zeros_like(sqdist)
    for i in range(int(kernel_num)):
        bw = base * (kernel_mul ** i)
        out = out + torch.exp(-sqdist / (bw + 1e-6))
        out = out / float(kernel_num)
    return out


def _mmd_rbf(
    source,
    target,
    kernel_mul=2.0,
    kernel_num=5,
    normalize_latent=True,
    max_samples_per_domain=0,
):
    if source is None or target is None:
        raise RuntimeError("source/target latent cannot be None for MMD")
    if source.dim() != 2 or target.dim() != 2:
        raise RuntimeError(f"MMD expects 2D latents, got source={tuple(source.shape)} target={tuple(target.shape)}")
    if source.size(1) != target.size(1):
        raise RuntimeError(f"MMD latent dim mismatch: source={tuple(source.shape)} target={tuple(target.shape)}")

    if normalize_latent:
        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

    # only cap MMD complexity; does not change supervised batch itself
    source = _subsample_latent_for_mmd(source, max_samples_per_domain=max_samples_per_domain)
    target = _subsample_latent_for_mmd(target, max_samples_per_domain=max_samples_per_domain)

    # pairwise squared distances without building [N, N, D]
    sq_xx = torch.cdist(source, source, p=2).pow(2)
    sq_yy = torch.cdist(target, target, p=2).pow(2)
    sq_xy = torch.cdist(source, target, p=2).pow(2)

    bandwidth = _estimate_mmd_bandwidth_from_sqdist(
        sq_xx.detach(),
        sq_yy.detach(),
        sq_xy.detach(),
    )

    k_xx = _multi_rbf_from_sqdist(
        sq_xx, bandwidth,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
    )
    k_yy = _multi_rbf_from_sqdist(
        sq_yy, bandwidth,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
    )
    k_xy = _multi_rbf_from_sqdist(
        sq_xy, bandwidth,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
    )

    loss = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return torch.clamp(loss, min=0.0)


def _coral_loss(source, target, normalize_latent=True):
    """Deep CORAL: squared Frobenius distance between source/target feature
    covariances. Uses the same domain-agnostic, window-mean latents as MMD."""
    if source is None or target is None:
        raise RuntimeError("source/target latent cannot be None for CORAL")
    if source.dim() != 2 or target.dim() != 2:
        raise RuntimeError(f"CORAL expects 2D latents, got source={tuple(source.shape)} target={tuple(target.shape)}")
    if source.size(1) != target.size(1):
        raise RuntimeError(f"CORAL latent dim mismatch: source={tuple(source.shape)} target={tuple(target.shape)}")
    if normalize_latent:
        # per-feature standardization using SOURCE stats (NOT row-wise L2:
        # row-normalising unit vectors collapses the covariance difference to ~0
        # once divided by 4*d*d). Standardizing keeps the covariance meaningful.
        mu  = source.mean(0, keepdim=True)
        std = source.std(0, keepdim=True).clamp(min=1e-6)
        source = (source - mu) / std
        target = (target - mu) / std
    d = int(source.size(1))
    ns, nt = int(source.size(0)), int(target.size(0))
    if ns < 2 or nt < 2:
        return source.new_zeros(())
    src_c = source - source.mean(dim=0, keepdim=True)
    tgt_c = target - target.mean(dim=0, keepdim=True)
    cov_s = (src_c.t() @ src_c) / float(ns - 1)
    cov_t = (tgt_c.t() @ tgt_c) / float(nt - 1)
    return (cov_s - cov_t).pow(2).sum() / (4.0 * float(d) * float(d))


def _dann_loss(discriminator, source, target, grl_lambda=1.0, normalize_latent=True):
    """DANN domain-adversarial loss. Discriminator separates source(0)/target(1);
    the GRL flips the gradient flowing into the encoder (scaled by grl_lambda) so
    the encoder learns domain-invariant features. Same latents as MMD/CORAL."""
    if discriminator is None:
        raise RuntimeError("DANN requires a domain_discriminator module")
    if source is None or target is None:
        raise RuntimeError("source/target latent cannot be None for DANN")
    if normalize_latent:
        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
    feat = torch.cat([source, target], dim=0)
    feat = _grad_reverse(feat, grl_lambda)
    logits = discriminator(feat).view(-1)
    labels = torch.cat([
        torch.zeros(int(source.size(0)), device=source.device, dtype=logits.dtype),
        torch.ones(int(target.size(0)), device=target.device, dtype=logits.dtype),
    ], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)

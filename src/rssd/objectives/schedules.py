"""Warm-up / ramp schedules for the auxiliary loss weights."""

from __future__ import annotations


__all__ = ["get_mmd_weight", "get_err_weight"]

def get_mmd_weight(epoch, w_max=0.05, warmup=5, ramp=10):
    if epoch <= warmup:
        return 0.0
    t = min(1.0, (epoch - warmup) / float(ramp))
    return w_max * t


def get_err_weight(epoch, w_max=0.2, warmup=5, ramp=10):
    if epoch <= warmup:
        return 0.0
    t = min(1.0, (epoch - warmup) / float(ramp))
    return w_max * t

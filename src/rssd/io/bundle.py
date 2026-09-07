"""Checkpoint bundle helpers: tolerant state-dict loading and weight averaging."""

from __future__ import annotations

import torch

__all__ = ["_load_state_dict_any"]

def _load_state_dict_any(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], ckpt
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"], ckpt
        # raw state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt, {"raw_state_dict": True}
    raise RuntimeError(f"Unrecognized checkpoint format: {path}")

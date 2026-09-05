"""Checkpoint bundle helpers: tolerant state-dict loading and weight averaging."""

from __future__ import annotations

import torch

__all__ = ["_load_state_dict_any", "_average_state_dicts"]

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


def _average_state_dicts(state_list):
    if not state_list:
        return None
    keys = list(state_list[0].keys())
    out = {}
    n = float(len(state_list))
    for k in keys:
        ref = state_list[0][k]
        if torch.is_tensor(ref) and ref.dtype.is_floating_point:
            acc = None
            for sd in state_list:
                t = sd[k].detach().to(dtype=torch.float32)
                acc = t.clone() if acc is None else (acc + t)
            out[k] = (acc / n).to(dtype=ref.dtype)
        else:
            # non-floating buffers / counters: keep the latest one
            out[k] = ref.detach().cpu().clone() if torch.is_tensor(ref) else ref
    return out

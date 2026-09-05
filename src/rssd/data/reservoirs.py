"""Reservoir ordering helpers shared by training and evaluation."""

from __future__ import annotations


__all__ = ["_build_idx_to_reservoir", "_get_reservoir_order", "_reservoir_order_from_encode_map"]

def _build_idx_to_reservoir(encode_map):
    """
    encode_map: {reservoir_name -> node_idx}
    return: {node_idx -> reservoir_name}
    """
    idx_to_reservoir = {}
    if encode_map is None:
        return idx_to_reservoir

    for name, idx in encode_map.items():
        idx = int(idx)
        if idx in idx_to_reservoir and idx_to_reservoir[idx] != name:
            raise ValueError(f"Duplicate node_idx={idx} for {idx_to_reservoir[idx]} vs {name}")
        idx_to_reservoir[idx] = name

    return idx_to_reservoir


def _get_reservoir_order(n_nodes, encode_map=None):
    order = [f"Node_{i}" for i in range(n_nodes)]
    if encode_map:
        for name, idx in encode_map.items():
            if 0 <= idx < n_nodes:
                order[idx] = name
    return order


def _reservoir_order_from_encode_map(encode_map, n_nodes: int):
    """encode_map: {name -> idx}. Return list[str] of length n_nodes in node order."""
    order = [None] * n_nodes
    if encode_map:
        for name, idx in encode_map.items():
            if 0 <= idx < n_nodes:
                order[idx] = name
    for i in range(n_nodes):
        if order[i] is None:
            order[i] = f"Node_{i}"
    return order

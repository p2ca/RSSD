"""A five-minute tour of the package, on synthetic tensors.

Reservoir records are not distributed with the code, so this example fabricates a batch with
the right shapes. It shows what the RSSD layer does to a dynamic state, that the variant
ladder is one configuration with switches rather than nine separate models, and how a real
run is launched.

    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rssd.config import MODEL_VARIANTS, build_train_cfg          # noqa: E402
from rssd.models.builder import build_model                      # noqa: E402
from rssd.utils import Data                                      # noqa: E402

TIN, PRED_LEN, N_NODES, N_FEATURES, HIDDEN = 30, 7, 5, 4, 32


def synthetic_window():
    """One forecast issue: `TIN` daily graphs, each carrying the four dynamic inputs."""
    edge_index = torch.zeros(2, 0, dtype=torch.long)             # no cross-reservoir edges
    return [Data(torch.randn(N_NODES, N_FEATURES), edge_index) for _ in range(TIN)]


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    torch.manual_seed(0)

    section("1. The forecast model")
    model = build_model(
        input_dim=N_FEATURES, hidden_dim=HIDDEN, num_layers=1, pred_len=PRED_LEN,
        dropout=0.0, latent_mode="attn", use_latent_proj=True, use_direct_head=True,
        num_reservoirs=N_NODES, use_reservoir_emb=True, reservoir_emb_dim=4,
        use_darsd=True, lcib_k=8, device="cpu").eval()

    graphs = synthetic_window()
    reservoir_ids = torch.arange(N_NODES)
    with torch.no_grad():
        prediction = model(graphs, reservoir_ids=reservoir_ids)
    print(f"{TIN} days x {N_NODES} reservoirs x {N_FEATURES} inputs "
          f"-> forecast {tuple(prediction.shape)} (reservoirs x lead days)")

    section("2. What RSSD does to the encoded state")
    with torch.no_grad():
        h_dyn = model.encode_latent(graphs, reservoir_ids=reservoir_ids, apply_darsd=False)
        h_shr, h_spc, assignments = model._lcib_decompose(h_dyn)
        h_rec = model._lcib_forward(h_dyn)
        gate = torch.sigmoid(model.lcib_gate)

    print(f"dynamic state h_dyn        {tuple(h_dyn.shape)}")
    print(f"  shared component h_shr   reconstructed on {assignments.shape[-1]} basis atoms")
    print(f"  site-specific h_spc      the remainder, h_dyn - h_shr")
    print(f"  the two sum back to h_dyn: max error "
          f"{float((h_shr + h_spc - h_dyn).abs().max()):.2e}")
    print(f"  recomposed h_rec = h_shr + (1 - g) h_spc, with gate g = {float(gate):.3f}: "
          f"max error {float((h_rec - (h_shr + (1 - gate) * h_spc)).abs().max()):.2e}")
    print(f"  assignment weights sum to one per window: "
          f"{float(assignments.sum(dim=-1).mean()):.3f}")

    section("3. The variant ladder is one configuration with switches")
    header = f"{'variant':26s} {'attributes':>10s} {'identity':>9s} {'RSSD':>5s} {'alignment':>10s}"
    print(header)
    for key in MODEL_VARIANTS:
        cfg = build_train_cfg(variant=key, dataset_tag="snow_source_v2", version_tag="v2")
        m = cfg["model"]
        attributes = m["use_res_static"] or m["use_meta_only_static"]
        print(f"{key:26s} {str(attributes):>10s} {str(m['use_reservoir_emb']):>9s} "
              f"{str(m['use_darsd']):>5s} {cfg['adaptation']['ALIGN_METHOD']:>10s}")

    section("4. Running the real thing")
    print("Point the package at a working copy of the data and the runs:")
    print("    export RSSD_DATA=/path/to/data")
    print("    export RSSD_LOGS=/path/to/runs")
    print()
    print("    python -m rssd.cli.train    --variant rssd_lstm --dataset snow_source_v2")
    print("    python -m rssd.cli.evaluate --scenario snow2rain --variant rssd_lstm")
    print()
    print("    python -m rssd.cli.train --list-runs        # every run in the manifest")


if __name__ == "__main__":
    main()

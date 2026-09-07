# RSSD

**Reservoir Shared–Specific Decomposition for multi-day reservoir inflow forecasting under
hydroclimatic regime shift.**

Forecasting inflow at a reservoir with a short usable record means borrowing from
reservoirs that have long ones. That works less well when the source and target sit in
different runoff-generation regimes — snowmelt-dominated against rainfall-driven — because
a sequence model pretrained in one regime carries response timing and scaling that do not
match the other.

RSSD addresses this inside the representation. The encoded dynamic state is reconstructed
on a learnable basis shared across the source reservoirs; the part the basis does not
explain is kept as a site-specific component and attenuated by a learned gate before the
two are recomposed. Reservoir identity enters *before* sequence encoding as a reservoir-ID
embedding, and reservoir attributes condition the recomposed state *after* the
decomposition, so the three sources of reservoir information play distinct roles in
transfer.

> Status: research code accompanying a journal manuscript in preparation. It extends the
> conference paper *Domain-Adaptive Reservoir Inflow Forecasting via Invariant-Basis
> Representation Decomposition* (ACM AI Leadership Summit 2026).

---

## Data

Inflow comes from the USACE reservoir-data platform maintained by the Nicholas Institute at
Duke University, and the meteorological drivers from the Daymet and Livneh products; all are
obtained from their providers rather than redistributed here. Point the package at your own
copy:

```bash
export RSSD_DATA=/path/to/data     # reservoirs_*.txt, meta/, parsed/, align/
export RSSD_LOGS=/path/to/runs     # checkpoints and evaluation outputs
```

Both default to `data/` and `logs/` inside the checkout.

## Install

```bash
pip install -e .
```

Python 3.10+ with PyTorch 2.1+. The published runs used Python 3.10, PyTorch 2.8 (CUDA 12.8)
on a single NVIDIA RTX 3060.

## Layout

```
src/rssd/
├── paths.py         resolve data/log locations from the environment
├── config.py        transfer scenarios, model variants, run configuration, manifest
├── profiles.py      the experiment profiles behind the variant ladder
├── metrics.py       correlation helpers, per-reservoir and per-lead-day NSE
├── data/            parsed tensors, scalers, static attributes
├── models/          Seq2SeqLSTM with the RSSD layer, checkpoint-locked model building
├── objectives/      alignment losses (MMD / CORAL / DANN) and the alignment-weight schedule
├── engine/          source training, target-history fine-tuning and evaluation
├── io/              checkpoint bundles and structured run outputs
└── cli/             command-line entry points

configs/experiments.yaml   which variants were trained on which sources, and the scenarios
data/reservoirs_*.txt      the frozen reservoir pools, in node order
tests/                     runs anywhere; data-dependent tests skip themselves
```

## Forecasting task and protocol

| | |
|---|---|
| Inputs | inflow, precipitation, maximum and minimum air temperature |
| Input window | 30 days |
| Forecast horizon | lead days 1–7 |
| Reservoirs | 33 (23 source, 10 target) |
| Static attributes | storage capacity, mean surface elevation, surface area, latitude, longitude, ground elevation |
| Target records | most recent 10 years, split chronologically 70 / 15 / 15 |
| Adaptation validation | the target's own validation block selects the adapted checkpoint |
| Split embargo | the leading 36 windows of the validation and test blocks are dropped (input window + horizon − 1) |

Six transfer scenarios: within-regime (`snow2snow`, `rain2rain`), cross-regime
(`snow2rain`, `rain2snow`) and mixed-source (`mixed2snow`, `mixed2rain`). Every scenario is
adapted on the target reservoir's own recent history before evaluation.

## Model variants

Each row adds one source of reservoir information, or exchanges one transfer mechanism,
holding everything else fixed.

| `--variant` | model | history | attributes | reservoir ID | alignment or decomposition |
|---|---|:---:|:---:|:---:|---|
| `seq2seq_lstm` | LSTM baseline | + | | | none |
| `attribute_informed_lstm` | LSTM + attributes | + | + | | none |
| `identity_informed_lstm` | LSTM + reservoir-ID embedding | + | | + | none |
| `fully_informed_lstm` | LSTM + attributes + reservoir-ID embedding | + | + | + | none |
| `fully_informed_mmd` | LSTM + attributes + reservoir-ID embedding + MMD | + | + | + | MMD on the complete feature vector |
| `fully_informed_coral` | LSTM + attributes + reservoir-ID embedding + CORAL | + | + | + | CORAL on the complete feature vector |
| `fully_informed_dann` | LSTM + attributes + reservoir-ID embedding + DANN | + | + | + | DANN on the complete feature vector |
| `rssd_lstm` | **RSSD (LSTM backbone)** | + | + | + | basis reconstruction with a gated residual |
| `rssd_transformer` | RSSD (Transformer backbone) | + | + | + | basis reconstruction with a gated residual |

The `--variant` keys are command-line identifiers and are unchanged; the model column gives the
name each variant carries in the manuscript.

`rssd_transformer` replaces both the recurrent encoder and its multi-horizon head with a Transformer encoder–decoder (`backbone="transformer_seq2seq"`); every lead day attends to the complete encoded history.

## Training a source model

```bash
python -m rssd.cli.train --variant rssd_lstm --dataset snow_source_v2 --version v2
```

The variant selects the rung of the ladder; the run configuration is derived from the frozen
defaults by applying that variant's profile, so the eight variants stay a controlled
comparison rather than eight hand-maintained configurations. Add `--print-config` to see the
resolved configuration without training, `--epochs` to cap the run, and `--output-dir` to
write elsewhere.

Checkpoints are written as `best_bundle.pt` and `avg_bundle.pt` (an average of the last
improved states) under `logs/train_<domain>/<run group>/<version>/`. A bundle carries the
architecture it was trained with, which is what evaluation reads back.

`configs/experiments.yaml` records the runs behind the reported comparison; both commands can
print them:

```bash
python -m rssd.cli.train --list-runs
python -m rssd.cli.evaluate --list-runs
```

## Evaluating a checkpoint

```bash
python -m rssd.cli.evaluate --scenario snow2rain --variant rssd_lstm
```

The command reads the architecture back out of the checkpoint rather than re-specifying it,
rebuilds the model for the target node set, restores the reservoir-ID embedding (or
initialises an unseen target set from the source reservoirs), adapts on the target's recent
history under the split protocol above, and writes `per_reservoir_r2_*.csv`,
`per_reservoir_daily_r2_*.csv`,
`metrics_summary_*.json`, `summary_row_*.csv` and a full run metadata record under
`logs/eval_<domain>/`.

Useful flags: `--no-finetune` (evaluate the source checkpoint directly), `--ckpt` (explicit checkpoint),
`--output-dir` (write elsewhere), `--device`.

## Tests

```bash
python -m unittest discover -s tests -t .
```

Tests that need the reservoir data skip themselves unless `RSSD_DATA` is set, so the
suite runs anywhere.

## Citation

A citation entry will be added when the manuscript is published. Until then, please cite the
conference paper.

## License

Released under the MIT License; see [LICENSE](LICENSE).

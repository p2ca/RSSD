# `data/`

What ships here: the frozen reservoir lists (`reservoirs_<pool>.txt`), one identifier per
line, in the node order the models and metrics are indexed by. They define the source and
target pools of the main partition and of the three alternative partitions. Also
[`sample/`](sample/README.md) — five **simulated** records with the same column layout,
enough to run preprocessing, training and evaluation without any of the data below.

What does not ship: the reservoir records themselves. Point `RSSD_DATA` at a directory
that holds them:

```
<RSSD_DATA>/
├── reservoirs_*.txt          # these files
├── align/<NIDID>.csv         # per-reservoir daily record: inflow, storage, elevation, meteorology
├── meta/                     # static reservoir attribute tables
└── parsed/<dataset tag>/     # windowed tensors and fitted scalers
```

`parsed/` is built from `align/` by `python -m rssd.cli.preprocess`; the required columns
are listed in `rssd.data.preprocess` and in [`sample/README.md`](sample/README.md).

Inflow and reservoir attributes come from the USACE reservoir-data platform maintained by
the Nicholas Institute at Duke University; precipitation and air temperature come from the
Daymet and Livneh products, and reservoir surface area from the National Inventory of Dams.

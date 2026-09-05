# `data/`

What ships here: the frozen reservoir lists (`reservoirs_<pool>.txt`), one identifier per
line, in the node order the models and metrics are indexed by. They define the source and
target pools of the main partition and of the three alternative partitions. Lists from
superseded protocol versions are not carried.

What does not ship: the reservoir records themselves and everything derived from them.
Point `RSSD_DATA` at a directory that holds them:

```
<RSSD_DATA>/
├── reservoirs_*.txt          # these files
├── align/<NIDID>.csv         # per-reservoir daily record: inflow, storage, elevation, meteorology
├── meta/                     # static reservoir attribute tables
└── parsed/<dataset tag>/     # windowed tensors and fitted scalers, built from the above
```

Inflow and reservoir attributes come from the USACE reservoir-data platform maintained by
the Nicholas Institute at Duke University; precipitation and air temperature come from the
Daymet and Livneh products. `scripts/fetch_nid_surface_area.py` assembles the surface-area
column of the attribute table from the National Inventory of Dams.

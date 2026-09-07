# `data/sample/` — a synthetic bundle for running the pipeline

**These records are simulated. They are not observations, and no number here should be
read as a property of a real reservoir.** They exist so that preprocessing, training and
evaluation can be run end to end without the reservoir database, and so that the column
contract the pipeline expects is documented by a file you can open.

The reservoir records behind the reported experiments are obtained from their providers
(see the manuscript) and are not redistributed here.

## What is in it

```
align/SYN000{1..5}.csv                    five simulated daily records, 2010-01-01 to 2021-12-31
meta/reservoir_latlon_elev_surface_area.csv   the six-attribute table for those five
reservoirs_sample_source.txt              SYN0001, SYN0002, SYN0003
reservoirs_sample_target.txt              SYN0004, SYN0005
generate_sample_data.py                   the generator; deterministic, same seed same files
```

The pools are sized so the geometry matches the real protocol closely enough to be
useful: the target pool yields 1818 adaptation windows against 1809 in the reported runs.

## The column contract

`align/<id>.csv`, one row per day:

| column | meaning |
|---|---|
| `date` | calendar date, parseable by `pandas.to_datetime` |
| `inflow` | reservoir inflow — the forecast target |
| `precip` | precipitation |
| `tmax` | maximum air temperature |
| `tmin` | minimum air temperature |
| `storage` | reservoir storage — read only when the static attributes are built |
| `elevation` | water-surface elevation — likewise |

`meta/reservoir_latlon_elev_surface_area.csv`, one row per reservoir:
`NIDID`, `LATITUDE`, `LONGITUDE`, `ELEV_M`, `SURFACE_AREA_KM2`.

Any directory with this layout can be used instead: point `--data-root` at it when
preprocessing, and `RSSD_DATA` at it when training and evaluating.

## Running the pipeline on it

```bash
python -m rssd.cli.preprocess --dataset sample_source --role source
python -m rssd.cli.preprocess --dataset sample_target --role target

export RSSD_DATA=$PWD/data/sample
python -m rssd.cli.train --variant rssd_lstm --dataset sample_source --version sample
python -m rssd.cli.evaluate --scenario sample2sample --variant rssd_lstm --version sample
```

The first two commands write `parsed/sample_{source,target}/`, which is ignored by git.

## Regenerating

```bash
python data/sample/generate_sample_data.py
```

Each record is an AR(1) baseflow recession around a seasonal level, with a decaying
response to simulated rainfall; storage integrates inflow against a constant release and
water-surface elevation follows storage. The five reservoirs differ in mean flow, in how
strongly they respond to rainfall, and in whether their seasonal peak is a spring melt or
a winter wet season.

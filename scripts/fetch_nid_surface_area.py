"""
Fetch National Inventory of Dams attributes for reservoirs in the local metadata table.

Outputs:
  - data/meta/reservoir_nid_attributes.csv
  - data/meta/reservoir_latlon_elev_surface_area.csv

The script is intentionally data-only. It does not change the active 5-D metadata
attribute-table contract.
"""

from __future__ import annotations

import csv
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_IN = PROJECT_ROOT / "data" / "meta" / "reservoir_latlon_elev.csv"
NID_ATTR_OUT = PROJECT_ROOT / "data" / "meta" / "reservoir_nid_attributes.csv"
AUGMENTED_OUT = PROJECT_ROOT / "data" / "meta" / "reservoir_latlon_elev_surface_area.csv"

NID_QUERY_URL = (
    "https://geospatial.sec.usace.army.mil/dls/rest/services/NID/"
    "National_Inventory_of_Dams_Public_Service/MapServer/0/query"
)

OUT_FIELDS = [
    "OBJECTID",
    "NIDID",
    "NAME",
    "STATE",
    "LATITUDE",
    "LONGITUDE",
    "IS_ASSOCIATED_STRUCTURE",
    "DATA_UPDATED",
    "SURFACE_AREA",
    "MAX_STORAGE",
    "NORMAL_STORAGE",
    "NID_STORAGE",
    "DRAINAGE_AREA",
]

ACRE_TO_KM2 = 0.0040468564224

# Local metadata contains NE05050 for Gavins Point Dam, while the current NID
# public service indexes the same dam as SD01094. Keep this explicit instead of
# relying on fuzzy name matching.
NIDID_ALIASES = {
    "NE05050": "SD01094",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_chunk(nidids: list[str]) -> list[dict[str, object]]:
    quoted = ",".join("'" + x.replace("'", "''") + "'" for x in nidids)
    params = {
        "where": f"NIDID IN ({quoted})",
        "outFields": ",".join(OUT_FIELDS),
        "returnGeometry": "false",
        "f": "json",
    }
    url = NID_QUERY_URL + "?" + urllib.parse.urlencode(params, safe=",()'")
    req = urllib.request.Request(url, headers={"User-Agent": "reservoir-metadata-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(f"NID query failed: {payload['error']}")
    return [feat["attributes"] for feat in payload.get("features", [])]


def fetch_by_name(name: str) -> list[dict[str, object]]:
    tokens = sorted(name_tokens(name), key=lambda x: (-len(x), x))
    if not tokens:
        return []
    phrase = " ".join(tokens[: min(2, len(tokens))]).upper()
    phrase_sql = phrase.replace("'", "''")
    params = {
        "where": f"upper(NAME) LIKE '%{phrase_sql}%'",
        "outFields": ",".join(OUT_FIELDS),
        "returnGeometry": "false",
        "f": "json",
    }
    url = NID_QUERY_URL + "?" + urllib.parse.urlencode(params, safe=",()'%")
    req = urllib.request.Request(url, headers={"User-Agent": "reservoir-metadata-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(f"NID name query failed for {name}: {payload['error']}")
    return [feat["attributes"] for feat in payload.get("features", [])]


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def name_tokens(name: str) -> set[str]:
    stop = {"dam", "dike", "saddle", "levee", "main", "north", "south", "east", "west"}
    clean = "".join(ch.lower() if ch.isalnum() else " " for ch in name)
    return {tok for tok in clean.split() if tok and tok not in stop}


def token_jaccard(a: str, b: str) -> float:
    ta = name_tokens(a)
    tb = name_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def distance_score(local: dict[str, str], rec: dict[str, object]) -> float:
    lat0 = as_float(local.get("LATITUDE"))
    lon0 = as_float(local.get("LONGITUDE"))
    lat1 = as_float(rec.get("LATITUDE"))
    lon1 = as_float(rec.get("LONGITUDE"))
    if None in (lat0, lon0, lat1, lon1):
        return 999.0
    return abs(lat0 - lat1) + abs(lon0 - lon1)


def pick_record(local: dict[str, str], candidates: list[dict[str, object]]) -> tuple[dict[str, object] | None, str]:
    if not candidates:
        return None, "missing_from_nid"

    local_name = str(local.get("NAME", ""))

    unique_surface = {
        as_float(rec.get("SURFACE_AREA"))
        for rec in candidates
        if as_float(rec.get("SURFACE_AREA")) is not None
    }

    if len(candidates) > 1 and len(unique_surface) == 1:
        selected = min(
            candidates,
            key=lambda rec: distance_score(local, rec) - token_jaccard(local_name, str(rec.get("NAME", ""))),
        )
        return selected, "duplicate_nidid_same_surface_area"

    def score(rec: dict[str, object]) -> tuple[float, float, int]:
        dist = distance_score(local, rec)
        name_sim = token_jaccard(local_name, str(rec.get("NAME", "")))
        assoc = str(rec.get("IS_ASSOCIATED_STRUCTURE", "")).strip().lower()
        assoc_penalty = 0 if assoc in {"no", "n", "false", "0", ""} else 1
        area_missing = 1 if as_float(rec.get("SURFACE_AREA")) is None else 0
        return (area_missing, assoc_penalty, dist - name_sim)

    selected = min(candidates, key=score)
    if len(candidates) == 1:
        note = "single_record"
    else:
        note = "duplicate_nidid_selected_by_area_association_distance_name"
    return selected, note


def pick_name_fallback(local: dict[str, str]) -> tuple[dict[str, object] | None, str]:
    candidates = fetch_by_name(str(local.get("NAME", "")))
    if not candidates:
        return None, "missing_from_nid"

    local_name = str(local.get("NAME", ""))
    filtered = []
    for rec in candidates:
        sim = token_jaccard(local_name, str(rec.get("NAME", "")))
        dist = distance_score(local, rec)
        if sim >= 0.5 and dist <= 0.1:
            filtered.append(rec)

    if not filtered:
        return None, "missing_from_nid"
    selected, note = pick_record(local, filtered)
    if selected is None:
        return None, "missing_from_nid"
    return selected, f"name_coordinate_fallback_to_nidid_{selected.get('NIDID')}"


def main() -> None:
    meta_rows = read_csv(META_IN)
    local_nidids = {
        str(row["NIDID"]).strip()
        for row in meta_rows
        if str(row.get("NIDID", "")).strip()
    }
    nidids = sorted(local_nidids | set(NIDID_ALIASES.values()))

    records: list[dict[str, object]] = []
    for group in chunks(nidids, 75):
        records.extend(fetch_chunk(group))
        time.sleep(0.15)

    by_nidid: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        by_nidid[str(rec.get("NIDID", "")).strip()].append(rec)

    attr_rows: list[dict[str, object]] = []
    augmented_rows: list[dict[str, object]] = []

    for local in meta_rows:
        nidid = str(local["NIDID"]).strip()
        candidates = by_nidid.get(nidid, [])
        selected, note = pick_record(local, candidates)
        if selected is None and nidid in NIDID_ALIASES:
            alias = NIDID_ALIASES[nidid]
            candidates = by_nidid.get(alias, [])
            selected, _alias_note = pick_record(local, candidates)
            if selected is not None:
                note = f"explicit_alias_to_nidid_{alias}"
        if selected is None:
            selected, note = pick_name_fallback(local)

        surface_acres = as_float(selected.get("SURFACE_AREA")) if selected else None
        surface_km2 = surface_acres * ACRE_TO_KM2 if surface_acres is not None else None

        attr = {
            "NIDID": nidid,
            "NIDID_SELECTED": selected.get("NIDID", "") if selected else "",
            "LOCAL_NAME": local.get("NAME", ""),
            "NID_NAME": selected.get("NAME", "") if selected else "",
            "LOCAL_STATE": local.get("STATE", ""),
            "NID_STATE": selected.get("STATE", "") if selected else "",
            "LOCAL_LATITUDE": local.get("LATITUDE", ""),
            "LOCAL_LONGITUDE": local.get("LONGITUDE", ""),
            "NID_LATITUDE": selected.get("LATITUDE", "") if selected else "",
            "NID_LONGITUDE": selected.get("LONGITUDE", "") if selected else "",
            "SURFACE_AREA_ACRES": surface_acres if surface_acres is not None else "",
            "SURFACE_AREA_KM2": surface_km2 if surface_km2 is not None else "",
            "MAX_STORAGE_ACREFT": selected.get("MAX_STORAGE", "") if selected else "",
            "NORMAL_STORAGE_ACREFT": selected.get("NORMAL_STORAGE", "") if selected else "",
            "NID_STORAGE_ACREFT": selected.get("NID_STORAGE", "") if selected else "",
            "DRAINAGE_AREA_SQMI": selected.get("DRAINAGE_AREA", "") if selected else "",
            "IS_ASSOCIATED_STRUCTURE": selected.get("IS_ASSOCIATED_STRUCTURE", "") if selected else "",
            "DATA_UPDATED": selected.get("DATA_UPDATED", "") if selected else "",
            "OBJECTID": selected.get("OBJECTID", "") if selected else "",
            "NID_MATCH_COUNT": len(candidates),
            "SELECTION_NOTE": note,
        }
        attr_rows.append(attr)

        aug = dict(local)
        aug["SURFACE_AREA_ACRES"] = attr["SURFACE_AREA_ACRES"]
        aug["SURFACE_AREA_KM2"] = attr["SURFACE_AREA_KM2"]
        aug["NID_MATCH_COUNT"] = attr["NID_MATCH_COUNT"]
        aug["NID_SELECTION_NOTE"] = attr["SELECTION_NOTE"]
        augmented_rows.append(aug)

    attr_fields = [
        "NIDID",
        "NIDID_SELECTED",
        "LOCAL_NAME",
        "NID_NAME",
        "LOCAL_STATE",
        "NID_STATE",
        "LOCAL_LATITUDE",
        "LOCAL_LONGITUDE",
        "NID_LATITUDE",
        "NID_LONGITUDE",
        "SURFACE_AREA_ACRES",
        "SURFACE_AREA_KM2",
        "MAX_STORAGE_ACREFT",
        "NORMAL_STORAGE_ACREFT",
        "NID_STORAGE_ACREFT",
        "DRAINAGE_AREA_SQMI",
        "IS_ASSOCIATED_STRUCTURE",
        "DATA_UPDATED",
        "OBJECTID",
        "NID_MATCH_COUNT",
        "SELECTION_NOTE",
    ]
    aug_fields = list(meta_rows[0].keys()) + [
        "SURFACE_AREA_ACRES",
        "SURFACE_AREA_KM2",
        "NID_MATCH_COUNT",
        "NID_SELECTION_NOTE",
    ]

    write_csv(NID_ATTR_OUT, attr_rows, attr_fields)
    write_csv(AUGMENTED_OUT, augmented_rows, aug_fields)

    missing = [row["NIDID"] for row in attr_rows if row["SURFACE_AREA_ACRES"] == ""]
    duplicate = [row["NIDID"] for row in attr_rows if int(row["NID_MATCH_COUNT"]) > 1]
    active_ids = []
    for name in [
        "reservoirs_snow_source.txt",
        "reservoirs_snow_target.txt",
        "reservoirs_rain_source.txt",
        "reservoirs_rain_target.txt",
    ]:
        active_ids.extend(
            x.strip()
            for x in (PROJECT_ROOT / "data" / name).read_text(encoding="utf-8").splitlines()
            if x.strip()
        )
    active_set = set(active_ids)
    active_missing = [rid for rid in active_ids if rid in missing]

    print(f"[ok] queried_nidids={len(nidids)} nid_records={len(records)}")
    print(f"[ok] wrote {NID_ATTR_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[ok] wrote {AUGMENTED_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[summary] missing_surface_area={len(missing)} active_missing_surface_area={len(active_missing)}")
    print(f"[summary] duplicate_nidid_records={len(duplicate)}")
    if active_missing:
        print("[active_missing] " + ", ".join(active_missing))


if __name__ == "__main__":
    main()

"""Shared processing: steps, downloads, tobac analysis, and the JSON the page reads."""
import json
import os
from datetime import timedelta, timezone

import numpy as np

from . import config, land, tracking
from .geo import lattice_index, offsets_km

UTC = timezone.utc
STEP_S = config.TIME_RES_MIN * 60


# ---------------------------------------------------------------- steps ----
class Steps:
    """Regular 5-minute steps for one disturbance with the centre at each step."""

    def __init__(self, track, start, end, extrap_h=0.0):
        t0 = int(np.ceil(start.timestamp() / STEP_S)) * STEP_S
        t1 = int(np.floor(end.timestamp() / STEP_S)) * STEP_S      # a step covers [t, t + 5 min)
        t = np.arange(t0, t1, STEP_S, dtype=np.int64)
        lat, lon, extrap = track.centre(t.astype("datetime64[s]"), extrap_h)
        ok = np.isfinite(lat)
        # keep the contiguous run of steps that have a centre
        if ok.any():
            first, last = np.argmax(ok), len(ok) - np.argmax(ok[::-1])
            t, lat, lon, extrap = t[first:last], lat[first:last], lon[first:last], extrap[first:last]
        else:
            t, lat, lon, extrap = t[:0], lat[:0], lon[:0], extrap[:0]
        self.t, self.lat, self.lon, self.extrap = t, lat, lon, extrap
        self.ci, self.cj = lattice_index(lat, lon, config.GRID_RES_DEG)

    def __len__(self):
        return len(self.t)


def missing(cache, steps, now, others=()):
    """Steps that still need downloading, newest first.

    Reuses steps held by other disturbances' caches (an invest that was
    renumbered, or two systems close together). Recent incomplete steps are
    asked for again because GLM files can arrive late.
    """
    need = []
    now_s = now.timestamp()
    for k in range(len(steps) - 1, -1, -1):
        t, ci, cj = int(steps.t[k]), int(steps.ci[k]), int(steps.cj[k])
        if cache.usable(t, ci, cj):
            if cache.complete(t) or now_s - t > 3 * 3600:
                continue
        elif any(cache.adopt(o, t, ci, cj) for o in others):
            continue
        need.append((t, float(steps.lat[k]), float(steps.lon[k])))
    return need


# ------------------------------------------------------------- analysis ----
def analyse(steps, cache, log=print):
    """tobac features near the centre. Returns (features DataFrame, info dict)."""
    half = int(round(config.TOBAC_HALF_DEG / config.GRID_RES_DEG))
    feats = tracking.detect(steps.t, steps.ci, steps.cj, cache, half, log=log)
    n_all = len(feats)
    feats = tracking.link(feats)
    feats = tracking.add_derived(feats)
    info = {"features_in_domain": int(n_all), "unsegmented": 0, "cells_near_land": 0}
    if n_all == 0:
        return feats.assign(x_km=[], y_km=[], r_km=[], land=[]), info

    # Land: flag whole cells that ever come within LAND_KM of a coast.
    feats["dist_to_land"] = land.distance_km(feats["lat"].to_numpy(), feats["lon"].to_numpy())
    near = (feats["dist_to_land"] <= config.LAND_KM) | feats["dist_to_land"].isna()
    bad_cells = set(feats.loc[near, "cell"].unique()) - {-1}
    feats["land"] = (near | feats["cell"].isin(bad_cells)).astype(int)
    info["cells_near_land"] = len(bad_cells)

    frame = feats["frame"].to_numpy()
    x, y, r = offsets_km(steps.lat[frame], steps.lon[frame], feats["lat"].to_numpy(), feats["lon"].to_numpy())
    feats["x_km"], feats["y_km"], feats["r_km"] = x, y, r
    feats = feats[feats["r_km"] <= config.RADIUS_KM]

    # Features whose segmentation failed have no statistics.
    bad = ~(feats["group_count_sum"] > 0)
    info["unsegmented"] = int(bad.sum())
    feats = feats[~bad].sort_values(["frame", "feature"]).reset_index(drop=True)
    return feats, info


# -------------------------------------------------------------- output -----
def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%MZ") if dt else None


def _r(values, nd):
    return [None if not np.isfinite(v) else (int(v) if nd == 0 else round(float(v), nd)) for v in values]


# ATCF fills the name column with a number word until a system is named
_PLACEHOLDER_NAMES = {"", "INVEST", "UNNAMED", "NONAME", "TEST"} | set(
    "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN SEVENTEEN "
    "EIGHTEEN NINETEEN TWENTY TWENTY-ONE TWENTY-TWO TWENTY-THREE TWENTY-FOUR TWENTY-FIVE TWENTY-SIX TWENTY-SEVEN "
    "TWENTY-EIGHT TWENTY-NINE THIRTY THIRTY-ONE THIRTY-TWO THIRTY-THREE THIRTY-FOUR THIRTY-FIVE".split()
    + "TWENTYONE TWENTYTWO TWENTYTHREE TWENTYFOUR TWENTYFIVE TWENTYSIX TWENTYSEVEN TWENTYEIGHT TWENTYNINE THIRTYONE".split())


def storm_label(track):
    num = track.number
    tag = f"{num}{'L' if track.basin == 'AL' else 'E'}" if track.is_invest and num.isdigit() else f"{track.basin}{num}"
    if track.is_invest:
        if not num.isdigit():   # archived invest, e.g. A6 -> 96
            tag = f"9{num[1]}{'L' if track.basin == 'AL' else 'E'}"
        return f"Invest {tag}"
    name = "" if track.name in _PLACEHOLDER_NAMES or track.name.startswith("GENESIS") else track.name.title()
    return f"{name} ({tag})" if name else tag


def build_product(key, track, steps, cache, feats, info, status, outcome, genesis, now, note=""):
    nf = np.array([cache.steps.get(int(t), (-1, 0, 0, 0))[0] for t in steps.t], dtype=int)
    sat = np.array([cache.steps.get(int(t), (0, 0, 0, 0))[3] for t in steps.t], dtype=int)
    change = np.flatnonzero(np.diff(sat, prepend=-999))
    extrap_from = int(np.argmax(steps.extrap)) if steps.extrap.any() else -1
    in_window = (track._t >= steps.t[0] - 6 * 3600) & (track._t <= steps.t[-1] + 6 * 3600) if len(steps) else np.zeros(len(track._t), bool)
    return {
        "id": key,
        "atcf_id": track.storm_id,
        "label": storm_label(track),
        "basin": track.basin,
        "year": track.year,
        "status": status,              # active | ended
        "outcome": outcome,            # invest | pre-genesis | developed | nondeveloper
        "genesis": _iso(genesis),
        "note": note,
        "updated": _iso(now),
        "method": config.method_summary(),
        "info": info,
        "track": {
            "t": [int(t) for t in track._t[in_window]],
            "lat": _r(track.lat[in_window], 2),
            "lon": _r(track.lon[in_window], 2),
            "vmax": _r(track.vmax[in_window], 0),
            "mslp": _r(track.mslp[in_window], 0),
            "type": [k for k, w in zip(track.kind, in_window) if w],
        },
        "steps": {
            "t0": int(steps.t[0]) if len(steps) else None,
            "dt": STEP_S,
            "n": len(steps),
            "clat": _r(steps.lat, 3),
            "clon": _r(steps.lon, 3),
            "nfiles": [int(v) for v in nf],          # -1 = not downloaded yet
            "expected_files": config.EXPECTED_FILES_PER_STEP,
            "sat": [[int(i), int(sat[i])] for i in change],
            "extrap_from": extrap_from,              # first step with an extrapolated centre
        },
        "features": {
            "k": [int(v) for v in feats["frame"]],
            "lat": _r(feats["lat"], 3),
            "lon": _r(feats["lon"], 3),
            "x": _r(feats["x_km"], 1),
            "y": _r(feats["y_km"], 1),
            "n": _r(feats["group_count_sum"], 0),
            "area": _r(feats["mean_group_area"], 1),       # km2
            "energy": _r(feats["mean_group_energy"], 2),   # fJ
            "cell": [int(v) for v in feats["cell"]],
            "life": _r(feats["lifetime_min"], 0),
            "land": [int(v) for v in feats["land"]],
            "px": _r(feats["ncells"], 0),
        },
    }


def index_entry(product):
    s, f, tr = product["steps"], product["features"], product["track"]
    t_end = s["t0"] + s["n"] * s["dt"] if s["n"] else None
    # hourly group counts over the last 72 h of the record (sea-only cells)
    hours = 72
    spark = [0] * hours
    if s["n"]:
        for k, n, landed in zip(f["k"], f["n"], f["land"]):
            if landed:
                continue
            age_h = int((s["n"] - 1 - k) * s["dt"] // 3600)
            if age_h < hours:
                spark[hours - 1 - age_h] += n
    sea = [n for n, landed in zip(f["n"], f["land"]) if not landed]
    return {
        "id": product["id"], "atcf_id": product["atcf_id"], "label": product["label"], "basin": product["basin"],
        "year": product["year"], "status": product["status"], "outcome": product["outcome"],
        "genesis": product["genesis"], "note": product["note"], "updated": product["updated"],
        "t_start": s["t0"], "t_end": t_end,
        "last": {"t": tr["t"][-1], "lat": tr["lat"][-1], "lon": tr["lon"][-1], "vmax": tr["vmax"][-1],
                 "mslp": tr["mslp"][-1], "type": tr["type"][-1]} if tr["t"] else None,
        "n_features": len(sea), "n_groups": int(sum(sea)), "spark": spark,
        "pending_steps": int(sum(1 for v in s["nfiles"] if v < 0)),
    }


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def load_index():
    path = os.path.join(config.DATA_DIR, "index.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"updated": None, "storms": []}


def save_index(index, now):
    index["updated"] = _iso(now)
    index["storms"].sort(key=lambda e: (e["status"] != "active", -(e["t_end"] or 0)))
    _write_json(os.path.join(config.DATA_DIR, "index.json"), index)


def publish(product, index):
    _write_json(os.path.join(config.DATA_DIR, "storms", product["id"] + ".json"), product)
    entry = index_entry(product)
    index["storms"] = [e for e in index["storms"] if e["id"] != product["id"]] + [entry]
    return entry


def remove(key, index):
    index["storms"] = [e for e in index["storms"] if e["id"] != key]
    for path in (os.path.join(config.DATA_DIR, "storms", key + ".json"), cache_path(key)):
        if os.path.exists(path):
            os.remove(path)


def cache_path(key):
    return os.path.join(config.CACHE_DIR, key + ".npz")


def window_for(track, now=None):
    """(start, end, genesis) of the analysis period for a finished disturbance."""
    genesis = track.genesis_time()
    if genesis is not None and not track.is_invest:
        start = max(track.first, genesis - timedelta(hours=config.LOOKBACK_H))
        return start, genesis, genesis
    end = track.last
    return max(track.first, end - timedelta(hours=config.MAX_LIFETIME_H)), end, None

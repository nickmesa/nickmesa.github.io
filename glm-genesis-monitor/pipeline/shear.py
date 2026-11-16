"""Deep and mid-level vertical wind shear around the centre of a record.

Follows the BEST_b*.nc files of the paper: the area-mean wind within 0-300 km of the center, upper level
minus lower level, for 850-200 hPa (deep) and 850-500 hPa (mid). Each 6-hourly analysis time gets one
vector per layer, kept as components (m s-1) of the shear vector (it points downshear).

Sources, best first:
  * ERA5 (NCAR copy on AWS), the same reanalysis as the vorticity. Published months behind real time.
  * GFS 0.25 degree analysis (NOAA open data on AWS), available a few hours after each analysis time.
A value from GFS is replaced by ERA5 once ERA5 has the hour.

    python -m pipeline.shear [--id al092026,ep052026] [--now 2026-10-11T06:00]
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from . import config, remote
from .geo import haversine

UTC = timezone.utc
LEVELS = (850, 500, 200)


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------- the maths ----
def area_mean(lat, lon, field, clat, clon, radius_km):
    """cos(latitude)-weighted mean of a field over the grid points within radius_km of a centre.

    lat, lon: 1-D grid coordinates (lon in -180..180); field: [lat, lon].
    """
    pad = radius_km / 111.0 + 0.5
    rows = np.flatnonzero(np.abs(lat - clat) <= pad)
    cols = np.flatnonzero(np.abs(lon - clon) <= pad / max(np.cos(np.radians(clat)), 0.2))
    if not (len(rows) and len(cols)):
        return None
    mlon, mlat = np.meshgrid(lon[cols], lat[rows])
    part = field[np.ix_(rows, cols)].astype(np.float64)
    inside = (haversine(clat, clon, mlat, mlon) <= radius_km) & np.isfinite(part)
    weight = np.cos(np.radians(mlat)) * inside
    return float((part * weight).sum() / weight.sum()) if inside.any() else None


def layer_shear(winds, lat, lon, clat, clon, radius_km=None):
    """winds: {(name 'u' or 'v', hPa): 2-D field}. Returns {'deep': (u, v), 'mid': (u, v)} in m s-1, or None."""
    radius_km = radius_km or config.SHEAR_RADIUS_KM
    mean = {}
    for key, field in winds.items():
        mean[key] = area_mean(lat, lon, field, clat, clon, radius_km)
    out = {}
    for name, (low, high) in config.SHEAR_LAYERS.items():
        parts = [mean.get(("u", high)), mean.get(("u", low)), mean.get(("v", high)), mean.get(("v", low))]
        if any(p is None for p in parts):
            return None
        out[name] = (parts[0] - parts[1], parts[2] - parts[3])
    return out


def magnitude_heading(u, v):
    """Speed and the compass direction the shear vector points toward (the downshear direction)."""
    return float(np.hypot(u, v)), float((np.degrees(np.arctan2(u, v)) + 360.0) % 360.0)


# ------------------------------------------------------------------ GFS ----
def gfs_url(when, base=None):
    return f"{base or config.GFS_URL}/gfs.{when:%Y%m%d}/{when:%H}/atmos/gfs.t{when:%H}z.pgrb2.0p25.f000"


def parse_idx(text):
    """{(variable, level text): (first byte, last byte or None)} from a GRIB2 .idx file."""
    rows = []
    for line in text.splitlines():
        p = line.split(":")
        if len(p) >= 5:
            rows.append((int(p[1]), p[3], p[4]))
    out = {}
    for i, (start, var, lev) in enumerate(rows):
        out[(var, lev)] = (start, rows[i + 1][0] - 1 if i + 1 < len(rows) else None)
    return out


def decode(message, clat, clon, half_deg):
    """Values of one GRIB2 message on a regular lat/lon grid, cropped around a centre. Returns lat, lon, field."""
    import eccodes

    gid = eccodes.codes_new_from_message(message)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        lat0, lon0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        dlat, dlon = eccodes.codes_get(gid, "jDirectionIncrementInDegrees"), eccodes.codes_get(gid, "iDirectionIncrementInDegrees")
        north_first = not eccodes.codes_get(gid, "jScansPositively")
        values = eccodes.codes_get_values(gid).reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    lat = lat0 + (-1 if north_first else 1) * dlat * np.arange(nj)
    lon = (lon0 + dlon * np.arange(ni) + 180.0) % 360.0 - 180.0
    pad_lon = half_deg / max(np.cos(np.radians(clat)), 0.2)
    rows = np.flatnonzero(np.abs(lat - clat) <= half_deg)
    dlon_to = (lon - clon + 180.0) % 360.0 - 180.0
    cols = np.flatnonzero(np.abs(dlon_to) <= pad_lon)
    return lat[rows], dlon_to[cols] + clon, values[np.ix_(rows, cols)]


def read_gfs(when, clat, clon, session=None, base=None):
    """Winds at the three levels from one GFS analysis, cropped around a centre. Raises FileNotFoundError if not posted."""
    session = session or requests.Session()
    url = gfs_url(when, base)
    r = session.get(url + ".idx", timeout=60)
    if r.status_code in (403, 404):
        raise FileNotFoundError(url)
    r.raise_for_status()
    idx = parse_idx(r.text)
    half = config.SHEAR_RADIUS_KM / 111.0 + 1.5
    winds, lat, lon = {}, None, None
    for var, name in (("UGRD", "u"), ("VGRD", "v")):
        for level in LEVELS:
            lo, hi = idx[(var, f"{level} mb")]
            rng = f"bytes={lo}-{hi}" if hi is not None else f"bytes={lo}-"
            g = session.get(url, headers={"Range": rng}, timeout=120)
            g.raise_for_status()
            lat, lon, winds[(name, level)] = decode(g.content, clat, clon, half)
    return lat, lon, winds


# ----------------------------------------------------------------- ERA5 ----
def era5_uv_url(day, name, base=None):
    stamp = f"{day:%Y%m%d}"
    code = {"u": "128_131_u", "v": "128_132_v"}[name]
    return f"{base or config.ERA5_URL}/e5.oper.an.pl/{day:%Y%m}/e5.oper.an.pl.{code}.{config.ERA5_UV_URL_SUFFIX}.{stamp}00_{stamp}23.nc"


def read_era5(when, clat, clon, base=None):
    """Winds at the three levels from the ERA5 analysis at one hour, cropped around a centre."""
    half = config.SHEAR_RADIUS_KM / 111.0 + 1.5
    day = when.replace(hour=0, minute=0, second=0, microsecond=0)
    winds, lat_out, lon_out = {}, None, None
    for name, vname in (("u", "U"), ("v", "V")):
        with remote.open_h5(era5_uv_url(day, name, base)) as (h5, raw):
            var = h5[vname] if vname in h5 else next(v for v in h5.values() if getattr(v, "ndim", 0) == 4)
            lat, lon = h5["latitude"][:].astype(np.float64), h5["longitude"][:].astype(np.float64)
            levels, hours = h5["level"][:].astype(np.float64), h5["time"][:].astype(np.int64)
            want = int((when - datetime(1900, 1, 1, tzinfo=UTC)).total_seconds() // 3600)
            hit = np.flatnonzero(hours == want)
            if not len(hit):
                raise FileNotFoundError(f"hour {when:%Y-%m-%d %H} not in the file")
            ks = [int(np.flatnonzero(levels == lv)[0]) for lv in LEVELS]
            lon180 = (lon + 180.0) % 360.0 - 180.0
            dlon_to = (lon180 - clon + 180.0) % 360.0 - 180.0
            rows = np.flatnonzero(np.abs(lat - clat) <= half)
            cols = np.flatnonzero(np.abs(dlon_to) <= half / max(np.cos(np.radians(clat)), 0.2))
            if not (len(rows) and len(cols)) or np.any(np.diff(cols) != 1) or np.any(np.diff(rows) != 1):
                raise ValueError("the crop wraps or lies outside the file")
            k0, k1 = min(ks), max(ks)
            block = var[int(hit[0]), k0:k1 + 1, int(rows[0]):int(rows[-1]) + 1, int(cols[0]):int(cols[-1]) + 1].astype(np.float32)
            for lv, k in zip(LEVELS, ks):
                winds[(name, lv)] = block[k - k0]
            lat_out, lon_out = lat[rows], dlon_to[cols] + clon
    return lat_out, lon_out, winds


# ------------------------------------------------------------ the record ----
def analysis_times(start, end, now):
    """6-hourly analysis times from the one at or before start to the one at or after end, not after now."""
    first = datetime.fromtimestamp(start, UTC).replace(minute=0, second=0, microsecond=0)
    first -= timedelta(hours=first.hour % config.SHEAR_STEP_H)
    times, when = [], first
    stop = datetime.fromtimestamp(end, UTC)
    while when <= stop + timedelta(hours=config.SHEAR_STEP_H):
        if when <= now:
            times.append(when)
        when += timedelta(hours=config.SHEAR_STEP_H)
    return times


def path_for(key, folder=None):
    return os.path.join(folder or config.SHEAR_DIR, key + ".json")


def load(key, folder=None):
    try:
        with open(path_for(key, folder)) as fh:
            return json.load(fh)
    except Exception:
        return None


def build(product, prior=None, now=None, say=log, gfs=None, era5=None):
    """The shear record for one lightning record, reusing what an earlier run stored.

    gfs, era5: readers (when, clat, clon) -> (lat, lon, winds); replaceable for tests.
    """
    from .env import Centre

    now = now or datetime.now(UTC)
    gfs = gfs or read_gfs
    era5 = era5 or read_era5
    centre = Centre(product)
    old = {}
    if prior:
        for i, t in enumerate(prior["t"]):
            old[t] = {k: prior[k][i] for k in ("src", "clat", "clon")}
            old[t].update({lay: (prior[lay]["u"][i], prior[lay]["v"][i]) for lay in config.SHEAR_LAYERS})
    era5_ok = bool(prior is None or (now.timestamp() - (prior.get("era5_try") or 0)) >= config.ERA5_RETRY_H * 3600)
    tried_era5 = False
    out = {}
    jobs = []
    for when in analysis_times(centre.start, centre.end, now):
        t = int(when.timestamp())
        clat, clon = centre(t)
        have = old.get(t)
        if have and haversine(have["clat"], have["clon"], clat, clon) <= config.SHEAR_REUSE_KM:
            if have["src"] == "era5":
                out[t] = have
                continue
            out[t] = have                    # a GFS value; try ERA5 below if it is old enough
        jobs.append((when, t, clat, clon))
    session = requests.Session()
    old_enough = timedelta(days=config.ERA5_MIN_AGE_DAYS)

    def one(job):
        when, t, clat, clon = job
        got = None
        if now - when >= old_enough and era5_ok:
            try:
                lat, lon, w = era5(when, clat, clon)
                s = layer_shear(w, lat, lon, clat, clon)
                got = ("era5", s) if s else None
            except Exception as err:
                say(f"    {when:%Y-%m-%d %HZ}: ERA5 not used ({type(err).__name__})")
            tried[0] = True
        if got is None and t not in out and now - when >= timedelta(hours=config.GFS_DELAY_H) and when >= config.GFS_FIRST:
            try:
                lat, lon, w = gfs(when, clat, clon)
                s = layer_shear(w, lat, lon, clat, clon)
                got = ("gfs", s) if s else None
            except Exception as err:
                say(f"    {when:%Y-%m-%d %HZ}: GFS not used ({type(err).__name__})")
        return t, clat, clon, got

    tried = [False]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for t, clat, clon, got in pool.map(one, jobs):
            if got:
                src, s = got
                out[t] = {"src": src, "clat": clat, "clon": clon, **s}
    ts = sorted(out)
    rec = {
        "id": product["id"], "radius_km": config.SHEAR_RADIUS_KM,
        "layers": {k: list(v) for k, v in config.SHEAR_LAYERS.items()},
        "t": ts, "src": [out[t]["src"] for t in ts],
        "clat": [round(out[t]["clat"], 2) for t in ts], "clon": [round(out[t]["clon"], 2) for t in ts],
        "t_end": product["steps"]["t0"] + product["steps"]["dt"] * product["steps"]["n"],
        "updated": now.strftime("%Y-%m-%dT%H:%MZ"),
        "era5_try": int(now.timestamp()) if tried[0] else (prior or {}).get("era5_try"),
    }
    for lay in config.SHEAR_LAYERS:
        rec[lay] = {"u": [round(out[t][lay][0], 2) for t in ts], "v": [round(out[t][lay][1], 2) for t in ts]}
    return rec


def needs_update(entry, have, now):
    """Whether an index entry's shear file should be (re)built."""
    if entry["status"] == "active" or not have:
        return True
    if have.get("t_end") != entry.get("t_end"):
        return True
    if all(s == "era5" for s in have["src"]):
        return False
    old = (now - datetime.fromtimestamp(entry["t_end"], UTC)) >= timedelta(days=config.ERA5_MIN_AGE_DAYS)
    due = now.timestamp() - (have.get("era5_try") or 0) >= config.ERA5_RETRY_H * 3600
    return old and due


def run(ids=None, now=None, folder=None, max_minutes=None, say=log, **readers):
    began = time.monotonic()
    now = now or datetime.now(UTC)
    folder = folder or config.SHEAR_DIR
    with open(os.path.join(config.DATA_DIR, "index.json")) as fh:
        storms = json.load(fh)["storms"]
    keep = {e["id"] for e in storms}
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.endswith(".json") and name[:-5] not in keep:
                os.remove(os.path.join(folder, name))            # left the lightning index
    todo, ended = [], 0
    for e in sorted(storms, key=lambda e: -(e.get("t_end") or 0)):
        if ids and e["id"] not in ids:
            continue
        have = load(e["id"], folder)
        if not needs_update(e, have, now):
            continue
        if e["status"] != "active" and not ids and not have:
            ended += 1
            if ended > config.SHEAR_ENDED_PER_RUN:
                continue
        todo.append(e["id"])
    for key in todo:
        if time.monotonic() - began > (max_minutes or config.SHEAR_MAX_MIN) * 60:
            say("  out of time; the rest is left for the next update")
            break
        try:
            with open(os.path.join(config.DATA_DIR, "storms", key + ".json")) as fh:
                product = json.load(fh)
            rec = build(product, load(key, folder), now=now, say=say, **readers)
        except Exception as err:
            say(f"{key}: FAILED {type(err).__name__}: {err}")
            continue
        say(f"{key}: {len(rec['t'])} times ({rec['src'].count('gfs')} GFS, {rec['src'].count('era5')} ERA5)")
        if rec["t"]:
            os.makedirs(folder, exist_ok=True)
            with open(path_for(key, folder) + ".tmp", "w") as fh:
                json.dump(rec, fh, separators=(",", ":"))
            os.replace(path_for(key, folder) + ".tmp", path_for(key, folder))
    return todo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", help="comma-separated record ids; default is every record that needs it")
    ap.add_argument("--now")
    ap.add_argument("--max-minutes", type=float)
    a = ap.parse_args(argv)
    now = datetime.fromisoformat(a.now).replace(tzinfo=UTC) if a.now else datetime.now(UTC)
    run(ids=set(a.id.split(",")) if a.id else None, now=now, max_minutes=a.max_minutes)
    return 0


if __name__ == "__main__":
    sys.exit(main())

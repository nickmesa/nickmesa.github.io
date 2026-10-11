"""GLM L2 LCFA files from the NOAA Open Data buckets: list, download, grid, cache.

Gridding follows nc_output.py: groups with group_quality_flag == 0 are binned
onto a 0.05 deg lat/lon grid over 5-minute periods, giving per cell the group
count, the summed optical energy and the mean group area.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import netCDF4
import numpy as np
import requests

from . import config
from .geo import lattice_index

UTC = timezone.utc
_SESSION = requests.Session()
_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=config.DOWNLOAD_THREADS * 2, max_retries=3)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)

SAT_CODE = {"noaa-goes16": 16, "noaa-goes17": 17, "noaa-goes18": 18, "noaa-goes19": 19}


# ----------------------------------------------------------- satellites ----
def bucket_for(when, lon):
    """Bucket holding the GLM that views this longitude at this time."""
    return _on_duty(config.WEST_SATS if lon < config.WEST_SWITCH_LON else config.EAST_SATS, when)


def _on_duty(fleet, when):
    chosen = fleet[0][1]
    for start, bucket in fleet:
        if when >= start:
            chosen = bucket
    return chosen


def other_view(bucket, when, lon):
    """Bucket of the satellite at the other position if it also views this longitude well, else None."""
    east = bucket in [b for _, b in config.EAST_SATS]
    if east and lon <= config.WEST_VIEW_EAST_LIMIT:
        return _on_duty(config.WEST_SATS, when)
    if not east and lon >= config.EAST_VIEW_WEST_LIMIT:
        return _on_duty(config.EAST_SATS, when)
    return None


def _alternates(bucket):
    """Other satellites that have held the same position (used if a listing is empty)."""
    for fleet in (config.EAST_SATS, config.WEST_SATS):
        names = [b for _, b in fleet]
        if bucket in names:
            return [b for b in reversed(names) if b != bucket]
    return []


# -------------------------------------------------------------- listing ----
def _list_prefix(bucket, prefix):
    base = config.GLM_URL.format(bucket=bucket)
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = _SESSION.get(base + "/", params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            return keys
        token = m.group(1)


def list_hour(bucket, hour):
    """[(start time, key)] of LCFA files whose scan starts in this UTC hour."""
    prefix = f"{config.GLM_PRODUCT}/{hour:%Y}/{hour:%j}/{hour:%H}/"
    out = []
    for key in _list_prefix(bucket, prefix):
        m = re.search(r"_s(\d{13})\d_e", key)
        if m:
            out.append((datetime.strptime(m.group(1), "%Y%j%H%M%S").replace(tzinfo=UTC), key))
    return sorted(out)


def list_hour_any(bucket, hour):
    """list_hour, falling back to the satellite's predecessor/successor if empty."""
    files = list_hour(bucket, hour)
    if files:
        return bucket, files
    for other in _alternates(bucket):
        files = list_hour(other, hour)
        if files:
            return other, files
    return bucket, []


# ------------------------------------------------------------- reading -----
def _fetch(bucket, key):
    url = f"{config.GLM_URL.format(bucket=bucket)}/{key}"
    for attempt in range(3):
        try:
            r = _SESSION.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 2:
                return None
    return None


def read_groups(raw):
    """Good-quality groups in one LCFA file: lat, lon, energy (J), area (m2)."""
    ds = netCDF4.Dataset("inmemory.nc", memory=raw)
    try:
        if len(ds.dimensions["number_of_groups"]) == 0:
            empty = np.empty(0)
            return empty, empty, empty, empty
        lat = np.ma.filled(ds.variables["group_lat"][:].astype(np.float64), np.nan)
        lon = np.ma.filled(ds.variables["group_lon"][:].astype(np.float64), np.nan)
        flag = np.ma.filled(ds.variables["group_quality_flag"][:], 1)
        energy = np.ma.filled(ds.variables["group_energy"][:].astype(np.float64), np.nan)
        area_var = ds.variables["group_area"]
        area = np.ma.filled(area_var[:].astype(np.float64), np.nan)
        # area units changed from km2 to m2 in 2019; work in m2
        if getattr(area_var, "units", "m2") == "km2":
            area = area * 1000.0**2
    finally:
        ds.close()
    good = (flag == 0) & np.isfinite(lat) & np.isfinite(lon)
    return lat[good], lon[good], energy[good], area[good]


def bin_groups(lat, lon, energy, area, ci, cj, half):
    """Bin groups onto the lattice cells within `half` cells of centre (ci, cj).

    Returns per occupied cell: gi, gj, count, energy sum, area sum, number of
    groups with a valid area (needed to rebuild the per-cell mean group area).
    """
    gi, gj = lattice_index(lat, lon, config.GRID_RES_DEG)
    keep = (np.abs(gi - ci) <= half) & (np.abs(gj - cj) <= half)
    if not keep.any():
        return None
    gi, gj, energy, area = gi[keep], gj[keep], energy[keep], area[keep]
    width = 2 * half + 1
    flat = (gi - (ci - half)) * width + (gj - (cj - half))
    cells, inv = np.unique(flat, return_inverse=True)
    n = np.bincount(inv, minlength=len(cells))
    e = np.bincount(inv, weights=np.nan_to_num(energy), minlength=len(cells))
    a = np.bincount(inv, weights=np.nan_to_num(area), minlength=len(cells))
    av = np.bincount(inv, weights=np.isfinite(area).astype(float), minlength=len(cells))
    return cells // width + (ci - half), cells % width + (cj - half), n, e, a, av


# --------------------------------------------------------------- cache -----
class CellCache:
    """Occupied grid cells for one disturbance, kept between runs.

    Cells live on the global 0.05 deg lattice, so a revised track or a renamed
    system (invest -> depression) can reuse them.
    """

    CELL_FIELDS = ("t", "gi", "gj", "n", "e", "a", "av")
    STEP_FIELDS = ("st", "s_nfiles", "s_ci", "s_cj", "s_sat")
    DTYPES = dict(t=np.int64, gi=np.int32, gj=np.int32, n=np.int32, e=np.float64, a=np.float64, av=np.int32,
                  st=np.int64, s_nfiles=np.int16, s_ci=np.int32, s_cj=np.int32, s_sat=np.int8)

    def __init__(self, path=None):
        self.path = path
        self.cells = {k: np.empty(0, self.DTYPES[k]) for k in self.CELL_FIELDS}
        self.steps = {}   # epoch seconds -> (nfiles, ci, cj, sat)
        self._new = []
        if path and os.path.exists(path):
            with np.load(path) as z:
                self.cells = {k: z[k] for k in self.CELL_FIELDS}
                self.steps = {int(t): (int(nf), int(ci), int(cj), int(s)) for t, nf, ci, cj, s in
                              zip(z["st"], z["s_nfiles"], z["s_ci"], z["s_cj"], z["s_sat"])}

    def usable(self, t, ci, cj):
        """True if step t is cached with a window that still covers centre (ci, cj)."""
        got = self.steps.get(int(t))
        if got is None:
            return False
        margin = int(round((config.CACHE_HALF_DEG - config.TOBAC_HALF_DEG) / config.GRID_RES_DEG))
        return abs(got[1] - ci) <= margin and abs(got[2] - cj) <= margin

    def complete(self, t):
        got = self.steps.get(int(t))
        return got is not None and got[0] >= config.EXPECTED_FILES_PER_STEP

    def put(self, t, nfiles, ci, cj, sat, parts):
        """Store one step. `parts` is a list of bin_groups() results for its files."""
        t = int(t)
        if t in self.steps:
            self.drop([t])
        self.steps[t] = (int(nfiles), int(ci), int(cj), int(sat))
        parts = [p for p in parts if p is not None]
        if not parts:
            return
        gi = np.concatenate([p[0] for p in parts])
        gj = np.concatenate([p[1] for p in parts])
        key = gi.astype(np.int64) * 10_000_000 + gj
        cells, inv = np.unique(key, return_inverse=True)
        sums = [np.bincount(inv, weights=np.concatenate([p[k] for p in parts]), minlength=len(cells)) for k in (2, 3, 4, 5)]
        self._new.append(dict(
            t=np.full(len(cells), t, np.int64), gi=(cells // 10_000_000).astype(np.int32),
            gj=(cells % 10_000_000).astype(np.int32), n=sums[0].astype(np.int32), e=sums[1], a=sums[2],
            av=sums[3].astype(np.int32)))

    def _flush(self):
        if self._new:
            for k in self.CELL_FIELDS:
                self.cells[k] = np.concatenate([self.cells[k]] + [d[k] for d in self._new])
            self._new = []
            order = np.argsort(self.cells["t"], kind="stable")
            self.cells = {k: v[order] for k, v in self.cells.items()}

    def drop(self, times):
        self._flush()
        times = np.asarray(list(times), dtype=np.int64)
        keep = ~np.isin(self.cells["t"], times)
        self.cells = {k: v[keep] for k, v in self.cells.items()}
        for t in times:
            self.steps.pop(int(t), None)

    def keep_only(self, times):
        wanted = set(int(t) for t in times)
        self.drop([t for t in self.steps if t not in wanted])

    def adopt(self, other, t, ci, cj):
        """Copy step t from another disturbance's cache if its window fits."""
        if not other.usable(t, ci, cj):
            return False
        other._flush()
        lo, hi = np.searchsorted(other.cells["t"], [int(t), int(t) + 1])
        nf, oci, ocj, sat = other.steps[int(t)]
        self.steps[int(t)] = (nf, oci, ocj, sat)
        if hi > lo:
            self._new.append({k: other.cells[k][lo:hi].copy() for k in self.CELL_FIELDS})
        return True

    def dense(self, times, ci, cj, half):
        """Dense (time, y, x) arrays on the storm-following grid.

        Returns group_count, total_energy (J) and total_area (m2), where
        total_area = mean group area * group count per cell as in
        tobac_single_day.py.
        """
        self._flush()
        width = 2 * half + 1
        count = np.zeros((len(times), width, width), np.float32)
        energy = np.zeros_like(count, dtype=np.float64)
        area = np.zeros_like(count, dtype=np.float64)
        c = self.cells
        for k, t in enumerate(times):
            lo, hi = np.searchsorted(c["t"], [int(t), int(t) + 1])
            if hi == lo:
                continue
            y = c["gi"][lo:hi] - (ci[k] - half)
            x = c["gj"][lo:hi] - (cj[k] - half)
            ok = (y >= 0) & (y < width) & (x >= 0) & (x < width)
            y, x = y[ok], x[ok]
            n = c["n"][lo:hi][ok]
            count[k, y, x] = n
            energy[k, y, x] = c["e"][lo:hi][ok]
            av = c["av"][lo:hi][ok]
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_area = np.where(av > 0, c["a"][lo:hi][ok] / av, 0.0)
            area[k, y, x] = mean_area * n
        return count, energy, area

    def save(self):
        self._flush()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        st = np.array(sorted(self.steps), dtype=np.int64)
        info = np.array([self.steps[int(t)] for t in st], dtype=np.int64).reshape(-1, 4)
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, st=st, s_nfiles=info[:, 0].astype(np.int16), s_ci=info[:, 1].astype(np.int32),
                            s_cj=info[:, 2].astype(np.int32), s_sat=info[:, 3].astype(np.int8), **self.cells)
        os.replace(tmp, self.path)


# ------------------------------------------------------------- download ----
def fill(requests_by_storm, log=print, now=None):
    """Download and grid every step that the caches are missing.

    requests_by_storm: {storm key: (cache, [(epoch_s, lat, lon), ...])} listing
    the steps each disturbance needs. Files are fetched once per satellite-hour
    and shared between disturbances that need them.
    """
    step = config.TIME_RES_MIN * 60
    half = int(round(config.CACHE_HALF_DEG / config.GRID_RES_DEG))
    plan = {}   # (bucket, hour) -> [(cache, t, ci, cj, lon)]
    for cache, wanted in requests_by_storm.values():
        for t, lat, lon in wanted:
            when = datetime.fromtimestamp(t, UTC)
            ci, cj = lattice_index(lat, lon, config.GRID_RES_DEG)
            plan.setdefault((bucket_for(when, lon), when.replace(minute=0, second=0, microsecond=0)), []).append(
                (cache, int(t), int(ci), int(cj), float(lon)))
    nfiles_total = 0
    settled = (now or datetime.now(UTC)).timestamp() - config.STANDIN_AFTER_H * 3600
    standins = {}   # (bucket, hour) -> {step: [keys]} listed from the other position
    for (bucket, hour) in sorted(plan, key=lambda k: (k[1], k[0])):
        jobs = plan[(bucket, hour)]
        used, listing = list_hour_any(bucket, hour)
        needed_t = sorted({j[1] for j in jobs})
        by_step = {t: [] for t in needed_t}
        for start, key in listing:
            t0 = int(start.timestamp()) // step * step
            if t0 in by_step:
                by_step[t0].append(key)
        # where each job's files come from: the usual satellite, or, for a period it
        # has nothing for, the satellite at the other position if that also sees the storm
        sources = []
        for cache, t, ci, cj, lon in jobs:
            src, keys = used, by_step[t]
            other = None if keys or t > settled else other_view(used, hour, lon)
            if other:
                if (other, hour) not in standins:
                    found = {}
                    for start, key in list_hour(other, hour):
                        found.setdefault(int(start.timestamp()) // step * step, []).append(key)
                    standins[(other, hour)] = found
                if standins[(other, hour)].get(t):
                    src, keys = other, standins[(other, hour)][t]
            sources.append((src, keys))
        wanted = sorted({(src, k) for src, keys in sources for k in keys})
        with ThreadPoolExecutor(config.DOWNLOAD_THREADS) as pool:
            raw = dict(zip(wanted, pool.map(lambda item: _fetch(*item), wanted)))
        nfiles_total += len(wanted)
        groups = {}
        for item, blob in raw.items():
            if blob is None:
                continue
            try:
                groups[item] = read_groups(blob)
            except Exception as err:   # a corrupt file should not stop the run
                log(f"  unreadable {item[1]}: {err}")
        for (cache, t, ci, cj, lon), (src, keys) in zip(jobs, sources):
            good = [(src, k) for k in keys if (src, k) in groups]
            parts = [bin_groups(*groups[g], ci, cj, half) for g in good]
            cache.put(t, len(good), ci, cj, SAT_CODE.get(src, 0), parts)
        borrowed = sorted({src for src, keys in sources if src != used and keys})
        log(f"  {used} {hour:%Y-%m-%d %HZ}: {len(wanted)} files for {len(needed_t)} steps"
            + (f" (some from {', '.join(borrowed)})" if borrowed else ""))
    return nfiles_total

"""Hourly ERA5 relative vorticity from NCAR's copy on AWS (s3://nsf-ncar-era5).

One file holds a day of one variable on all 37 pressure levels, stored as one
compressed chunk per hour. A needed hour is read with a single range request,
reduced to the two layer means used in the paper (area_averaged_vort_series()
in mesa_functions.py: low levels 900-700 hPa, mid levels 600-400 hPa) over the
western hemisphere, and kept on disk so storms that overlap in time share it.

This module must not import netCDF4: read_day() runs in worker processes that
load h5py.
"""
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import config, remote
from .geo import destination, haversine

UTC = timezone.utc
SCALE = 1e5                    # fields are held in units of 1e-5 s-1


def day_url(day, base=None):
    stamp = f"{day:%Y%m%d}"
    return f"{base or config.ERA5_URL}/e5.oper.an.pl/{day:%Y%m}/e5.oper.an.pl.128_138_vo.ll025sc.{stamp}00_{stamp}23.nc"


def cache_file(when, folder=None):
    return os.path.join(folder or os.path.join(config.CACHE_DIR, "era5"), f"{when:%Y%m%d%H}.npz")


def _layer(levels, bounds):
    lo, hi = min(bounds), max(bounds)
    return np.flatnonzero((levels >= lo) & (levels <= hi))


def read_day(job):
    """Read some hours of one day's file and save the layer means. Runs in a worker process.

    job: dict(url, day (ISO date), hours [0-23], region (south, north, west, east), low, mid, folder)
    """
    out = {"day": job["day"], "ok": [], "error": "", "missing": False, "requests": 0, "bytes": 0, "seconds": 0.0}
    began = time.monotonic()
    try:
        day = datetime.fromisoformat(job["day"]).replace(tzinfo=UTC)
        south, north, west, east = job["region"]
        with remote.open_h5(job["url"]) as (h5, raw):
            var = h5["VO"] if "VO" in h5 else next(v for v in h5.values() if getattr(v, "ndim", 0) == 4)
            lat, lon = h5["latitude"][:].astype(np.float64), h5["longitude"][:].astype(np.float64)
            levels, hours = h5["level"][:].astype(np.float64), h5["time"][:].astype(np.int64)
            lon180 = (lon + 180.0) % 360.0 - 180.0
            rows = np.flatnonzero((lat >= south) & (lat <= north))
            cols = np.flatnonzero((lon180 >= west) & (lon180 < east))
            low, mid = _layer(levels, job["low"]), _layer(levels, job["mid"])
            if not (len(rows) and len(cols) and len(low) and len(mid)):
                raise ValueError("the file does not cover the region or the pressure layers")
            k0, k1 = int(min(low.min(), mid.min())), int(max(low.max(), mid.max()))
            if np.any(np.diff(cols) != 1):
                raise ValueError("the region wraps around the file's longitude seam")
            fill = np.asarray(var.attrs["_FillValue"]).ravel()[0] if "_FillValue" in var.attrs else None
            since = datetime(1900, 1, 1, tzinfo=UTC)
            out["chunks"] = list(var.chunks) if var.chunks else None
            out["levels_low"] = [float(v) for v in levels[low]]
            out["levels_mid"] = [float(v) for v in levels[mid]]
            os.makedirs(job["folder"], exist_ok=True)
            for hour in job["hours"]:
                when = day + timedelta(hours=hour)
                want = int((when - since).total_seconds() // 3600)
                hit = np.flatnonzero(hours == want)
                if not len(hit):
                    continue
                block = var[int(hit[0]), k0:k1 + 1, int(rows[0]):int(rows[-1]) + 1, int(cols[0]):int(cols[-1]) + 1].astype(np.float32)
                if fill is not None:
                    block[block == fill] = np.nan
                path = cache_file(when, job["folder"])
                np.savez_compressed(path + ".tmp.npz", low=(block[low - k0].mean(axis=0) * SCALE).astype(np.float16),
                                    mid=(block[mid - k0].mean(axis=0) * SCALE).astype(np.float16),
                                    lat=lat[rows].astype(np.float32), lon=lon180[cols].astype(np.float32))
                os.replace(path + ".tmp.npz", path)
                out["ok"].append(hour)
            out["requests"], out["bytes"] = raw.requests, raw.bytes
    except FileNotFoundError:
        out["missing"] = True          # NCAR has not published this day yet
    except Exception as err:
        out["error"] = f"{type(err).__name__}: {err}"[:200]
    out["seconds"] = round(time.monotonic() - began, 1)
    return out


def jobs_for(hours):
    """read_day() jobs for the hours that are not on disk yet."""
    by_day = {}
    for when in hours:
        if not os.path.exists(cache_file(when)):
            by_day.setdefault(when.date(), []).append(when.hour)
    folder = os.path.join(config.CACHE_DIR, "era5")
    return [{"url": day_url(day), "day": day.isoformat(), "hours": sorted(set(hs)), "region": config.ERA5_REGION,
             "low": config.VORT_LOW_HPA, "mid": config.VORT_MID_HPA, "folder": folder}
            for day, hs in sorted(by_day.items())]


_HELD = {}


def _field(when):
    path = cache_file(when)
    if path not in _HELD:
        if not os.path.exists(path):
            return None
        if len(_HELD) > 6:
            _HELD.clear()
        with np.load(path) as z:
            _HELD[path] = {k: z[k].astype(np.float32) for k in ("low", "mid", "lat", "lon")}
    return _HELD[path]


def sample(when, clat, clon):
    """Layer-mean vorticity around a centre at one hour, or None if ERA5 does not have it.

    Returns dict(low, mid: n x n arrays on the storm-centred grid, 1e-5 s-1;
    mean_low, mean_mid: area means within RADIUS_KM, weighted by cos(latitude)
    over the ERA5 grid points inside the circle, as in the paper code).
    """
    from scipy.ndimage import map_coordinates

    f = _field(when)
    if f is None:
        return None
    lat, lon = f["lat"].astype(np.float64), f["lon"].astype(np.float64)
    n = int(round(2 * config.VORT_HALF_KM / config.VORT_RES_KM)) + 1
    edge = np.linspace(-config.VORT_HALF_KM, config.VORT_HALF_KM, n)
    gx, gy = np.meshgrid(edge, -edge)
    glat, glon = destination(clat, clon, gx, gy)
    fi = (glat - lat[0]) / (lat[1] - lat[0])
    fj = (glon - lon[0]) / (lon[1] - lon[0])
    out = {}
    for name in ("low", "mid"):
        grid = map_coordinates(f[name], [fi, fj], order=1, mode="constant", cval=np.nan)
        out[name] = grid.astype(np.float32)
    # area mean on the native grid
    pad = config.RADIUS_KM / 111.0 + 0.5
    rows = np.flatnonzero(np.abs(lat - clat) <= pad)
    cols = np.flatnonzero(np.abs(lon - clon) <= pad / max(np.cos(np.radians(clat)), 0.2))
    if len(rows) and len(cols):
        mlon, mlat = np.meshgrid(lon[cols], lat[rows])
        inside = haversine(clat, clon, mlat, mlon) <= config.RADIUS_KM
        weight = np.cos(np.radians(mlat)) * inside
        for name in ("low", "mid"):
            part = f[name][np.ix_(rows, cols)].astype(np.float64)
            ok = inside & np.isfinite(part)
            out["mean_" + name] = float((part[ok] * weight[ok]).sum() / weight[ok].sum()) if ok.any() else None
    else:
        out["mean_low"] = out["mean_mid"] = None
    return out

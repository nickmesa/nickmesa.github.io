"""ABI channel 13 brightness temperature around a storm centre.

Follows Get_ABI.py / nc_output.py in the paper code: L1b full-disk radiances
are converted to brightness temperature with the Planck constants in the file,
pixels with a data quality flag of 1 or more or a temperature outside
150-350 K are discarded, and the image is resampled by nearest neighbour.
Here the target is a storm-centred grid in km instead of a lat/lon box, and
only the part of each file that covers it is read (see remote.py).

This module must not import netCDF4 (directly or through glm.py): it runs in
worker processes that load h5py.
"""
import re
from datetime import datetime, timezone

import numpy as np
import requests

from . import config, remote
from .geo import centred_grid, destination

UTC = timezone.utc
SAT_CODE = {"noaa-goes16": 16, "noaa-goes17": 17, "noaa-goes18": 18, "noaa-goes19": 19}
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=2)
        _SESSION.mount("https://", adapter)
        _SESSION.mount("http://", adapter)
    return _SESSION


# ----------------------------------------------------------- satellites ----
def _on_duty(fleet, when):
    chosen = fleet[0][1]
    for start, bucket in fleet:
        if when >= start:
            chosen = bucket
    return chosen


def buckets_for(when, lon):
    """Buckets to try for a scan, best first: the satellite used for the lightning, then the other position."""
    east, west = _on_duty(config.EAST_SATS, when), _on_duty(config.WEST_SATS, when)
    if lon < config.WEST_SWITCH_LON:
        return [west] + ([east] if lon >= config.ABI_EAST_VIEW_WEST_LIMIT else [])
    return [east] + ([west] if lon <= config.ABI_WEST_VIEW_EAST_LIMIT else [])


# -------------------------------------------------------------- listing ----
def list_hour(bucket, hour, base=None):
    """[(scan start, key)] of channel 13 full-disk files whose scan starts in this UTC hour."""
    base = (base or config.GLM_URL).format(bucket=bucket)
    prefix = f"{config.ABI_PRODUCT}/{hour:%Y}/{hour:%j}/{hour:%H}/"
    want = re.compile(rf"C{config.ABI_CHANNEL:02d}_G\d\d_s(\d{{13}})\d_e")
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = _session().get(base + "/", params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    out = []
    for key in keys:
        m = want.search(key)
        if m:
            out.append((datetime.strptime(m.group(1), "%Y%j%H%M%S").replace(tzinfo=UTC), key))
    return sorted(out)


# ----------------------------------------------------------- navigation ----
def scan_angles(lat, lon, lon0, h, req, rpol):
    """ABI fixed-grid scan angles (x east-west, y north-south, radians) of points on the Earth.

    GOES-R Product User Guide, section 4.2.8.2. Also returns whether each
    point is on the visible side of the Earth.
    """
    phi, lam = np.radians(lat), np.radians(lon) - np.radians(lon0)
    H = h + req
    e2 = 1.0 - (rpol / req) ** 2
    phic = np.arctan((rpol / req) ** 2 * np.tan(phi))
    rc = rpol / np.sqrt(1.0 - e2 * np.cos(phic) ** 2)
    sx = H - rc * np.cos(phic) * np.cos(lam)
    sy = -rc * np.cos(phic) * np.sin(lam)
    sz = rc * np.sin(phic)
    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    visible = H * (H - sx) >= sy ** 2 + (req / rpol) ** 2 * sz ** 2
    return x, y, visible


def _attr(obj, name, default=None):
    if name not in obj.attrs:
        return default
    value = np.asarray(obj.attrs[name]).ravel()[0]
    return value.decode() if isinstance(value, bytes) else value


def _packed(dataset, box):
    """Unpack a CF-packed integer variable: (float values, raw integers)."""
    raw = dataset[box]
    if str(_attr(dataset, "_Unsigned", "")).lower() == "true":
        raw = raw.view(raw.dtype.str.replace("i", "u"))
    values = raw.astype(np.float32) * np.float32(_attr(dataset, "scale_factor", 1.0)) + np.float32(_attr(dataset, "add_offset", 0.0))
    fill = _attr(dataset, "_FillValue")
    if fill is not None:
        fill = np.asarray(fill).astype(dataset.dtype).view(raw.dtype)
        values[raw == fill] = np.nan
    return values, raw


# --------------------------------------------------------------- frames ----
def read_frame(job):
    """Brightness temperature on the storm-centred grid for one scan. Runs in a worker process.

    job: dict(url, clat, clon, half_km, res_km, valid, radius_km, thresholds, max_bad, tb_max, tb_step)
    Returns a dict with the image as bytes (uint8, 0 = missing or warmest), the
    fraction of good pixels, the cold cloud fractions, and what the read cost.
    """
    out = {"url": job["url"], "error": "", "requests": 0, "bytes": 0}
    try:
        gx, gy = centred_grid(job["half_km"], job["res_km"])
        lat, lon = destination(job["clat"], job["clon"], gx, gy)
        with remote.open_h5(job["url"], session=_session()) as (h5, raw):
            proj = h5["goes_imager_projection"]
            x, y, visible = scan_angles(lat, lon, float(_attr(proj, "longitude_of_projection_origin")),
                                        float(_attr(proj, "perspective_point_height")),
                                        float(_attr(proj, "semi_major_axis")), float(_attr(proj, "semi_minor_axis")))
            rad = h5["Rad"]
            ny, nx = rad.shape
            col = np.rint((x - float(_attr(h5["x"], "add_offset"))) / float(_attr(h5["x"], "scale_factor"))).astype(np.int64)
            row = np.rint((y - float(_attr(h5["y"], "add_offset"))) / float(_attr(h5["y"], "scale_factor"))).astype(np.int64)
            inside = visible & (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
            tb = np.full(lat.shape, np.nan, dtype=np.float32)
            if inside.any():
                r0, r1, c0, c1 = row[inside].min(), row[inside].max(), col[inside].min(), col[inside].max()
                box = (slice(int(r0), int(r1) + 1), slice(int(c0), int(c1) + 1))
                radiance, _ = _packed(rad, box)
                dqf = h5["DQF"][box]
                fk1, fk2 = float(h5["planck_fk1"][()]), float(h5["planck_fk2"][()])
                bc1, bc2 = float(h5["planck_bc1"][()]), float(h5["planck_bc2"][()])
                with np.errstate(invalid="ignore", divide="ignore"):
                    temp = (fk2 / np.log(fk1 / radiance + 1.0) - bc1) / bc2
                temp[(dqf != 0) | ~(temp >= job["valid"][0]) | ~(temp <= job["valid"][1])] = np.nan
                tb[inside] = temp[row[inside] - r0, col[inside] - c0]
            out["requests"], out["bytes"] = raw.requests, raw.bytes
            out["chunks"] = list(rad.chunks) if rad.chunks else None
        good = np.isfinite(tb)
        out["valid"] = round(float(good.mean()), 3)
        within = np.hypot(gx, gy) <= job["radius_km"]
        bad = 1.0 - float(good[within].mean())
        if bad > job["max_bad"]:
            out["ccf"] = None
        else:
            # as cold_cloud_fraction(): cold pixels over all pixels within the radius
            out["ccf"] = [round(float((tb[within] <= k).sum()) / int(within.sum()), 4) for k in job["thresholds"]]
        scaled = np.rint((job["tb_max"] - np.where(good, tb, job["tb_max"])) / job["tb_step"])
        out["image"] = np.clip(scaled, 0, 255).astype(np.uint8).tobytes()
        out["tb_min"] = round(float(np.nanmin(tb)), 1) if good.any() else None
    except FileNotFoundError as err:
        out["error"] = f"missing: {err}"
    except Exception as err:      # one bad file must not stop a case
        out["error"] = f"{type(err).__name__}: {err}"[:200]
    return out


def frame_job(url, clat, clon):
    """The arguments read_frame() needs, taken from config in the parent process."""
    return {"url": url, "clat": float(clat), "clon": float(clon), "half_km": config.IR_HALF_KM, "res_km": config.IR_RES_KM,
            "valid": config.IR_TB_VALID, "radius_km": config.RADIUS_KM, "thresholds": list(config.CCF_THRESHOLDS_K),
            "max_bad": config.IR_MAX_BAD, "tb_max": config.IR_TB_MAX, "tb_step": config.IR_TB_STEP}

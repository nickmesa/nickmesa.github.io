"""The environment of an archived case: channel 13 IR and ERA5 vorticity.

For one record in the archive this writes, into config.ENV_DIR:

    <id>.json       frame times, cold cloud fractions, hourly vorticity grids
                    and their 200 km means
    <id>_<k>.webp   the IR frames, IR_SHEET_FRAMES to a sheet

Everything is on storm-centred grids (x east, y north of the interpolated
centre), so it lines up with the lightning feature offsets.
"""
import atexit
import base64
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

from . import abi, config, era5

UTC = timezone.utc
_POOLS = {}


def _pool(kind, workers):
    """A process pool kept for the whole run (workers load h5py; see remote.py)."""
    if workers <= 0:
        return None
    if kind not in _POOLS:
        _POOLS[kind] = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    return _POOLS[kind]


@atexit.register
def _close_pools():
    for pool in _POOLS.values():
        pool.shutdown(wait=False, cancel_futures=True)
    _POOLS.clear()


def _map(kind, workers, fn, jobs):
    pool = _pool(kind, workers)
    if pool is None:
        return [fn(j) for j in jobs]
    return list(pool.map(fn, jobs, chunksize=max(1, len(jobs) // (workers * 8))))


def _dt(seconds):
    return datetime.fromtimestamp(int(seconds), UTC)


class Centre:
    """Interpolated centre of a record at any time inside (or just outside) its window."""

    def __init__(self, product):
        s = product["steps"]
        self.t = s["t0"] + s["dt"] * np.arange(s["n"], dtype=np.float64)
        self.lat, self.lon = np.asarray(s["clat"], float), np.asarray(s["clon"], float)
        self.start, self.end = int(s["t0"]), int(s["t0"] + s["dt"] * s["n"])

    def __call__(self, when):
        return float(np.interp(when, self.t, self.lat)), float(np.interp(when, self.t, self.lon))


def find_scans(centre, log=print):
    """The channel 13 scans a record needs, oldest first.

    Each is dict(t, bucket, key, alt): `alt` is the same scan from the other
    satellite position where that also sees the storm, used if the first
    choice turns out to be missing or mostly bad. One scan before the window
    is kept so the first minutes have an image.
    """
    first = _dt(centre.start) - timedelta(minutes=20)
    hour = first.replace(minute=0, second=0, microsecond=0)
    scans, listings = [], {}

    def listing(bucket, hour):
        if (bucket, hour) not in listings:
            try:
                listings[(bucket, hour)] = abi.list_hour(bucket, hour)
            except Exception as err:
                log(f"  could not list {bucket} {hour:%Y-%m-%d %HZ}: {err}")
                listings[(bucket, hour)] = []
        return listings[(bucket, hour)]

    while hour.timestamp() <= centre.end:
        mid = min(max(hour.timestamp() + 1800, centre.start), centre.end)
        choices = [(bucket, listing(bucket, hour)) for bucket in abi.buckets_for(hour, centre(mid)[1])]
        choices = [c for c in choices if c[1]]
        if choices:
            bucket, found = choices[0]
            other = choices[1] if len(choices) > 1 else None
            for start, key in found:
                alt = None
                if other:
                    near = min(other[1], key=lambda f: abs((f[0] - start).total_seconds()))
                    if abs((near[0] - start).total_seconds()) <= 300:
                        alt = (other[0], near[1])
                scans.append({"t": int(start.timestamp()), "bucket": bucket, "key": key, "alt": alt})
        hour += timedelta(hours=1)
    scans = sorted((s for s in scans if s["t"] <= centre.end), key=lambda s: s["t"])
    before = [s for s in scans if s["t"] <= centre.start]
    return before[-1:] + [s for s in scans if s["t"] > centre.start]


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def write(key, centre, frames, hours, samples, now, levels=None):
    """Write <key>.json and its image sheets. Returns (document, bytes of the sheets).

    frames: [dict(t, sat, valid, ccf, image)] with `image` the px*px bytes of
    one scan; samples: one era5.sample() result or None per hour.
    """
    from PIL import Image

    os.makedirs(config.ENV_DIR, exist_ok=True)
    px = int(round(2 * config.IR_HALF_KM / config.IR_RES_KM))
    per, cols = config.IR_SHEET_FRAMES, config.IR_SHEET_COLS
    sheets, ir_bytes = [], 0
    for k in range(0, len(frames), per):
        part = frames[k:k + per]
        rows = -(-len(part) // cols)
        sheet = np.zeros((rows * px, cols * px), dtype=np.uint8)
        for i, f in enumerate(part):
            y, x = (i // cols) * px, (i % cols) * px
            sheet[y:y + px, x:x + px] = np.frombuffer(f["image"], dtype=np.uint8).reshape(px, px)
        name = f"{key}_{k // per}.webp"
        Image.fromarray(sheet, mode="L").save(os.path.join(config.ENV_DIR, name), "WEBP", quality=config.IR_QUALITY, method=4)
        ir_bytes += os.path.getsize(os.path.join(config.ENV_DIR, name))
        sheets.append(name)

    n = int(round(2 * config.VORT_HALF_KM / config.VORT_RES_KM)) + 1
    grids = {name: np.zeros((len(hours), n, n), dtype=np.int8) for name in ("low", "mid")}
    means = {"low": [], "mid": []}
    have = [int(s is not None) for s in samples]
    for i, s in enumerate(samples):
        for name in ("low", "mid"):
            if s is None:
                means[name].append(None)
                continue
            scaled = np.nan_to_num(s[name] * 1e-5 / config.VORT_UNIT, nan=0.0)
            grids[name][i] = np.clip(np.rint(scaled), -127, 127).astype(np.int8)
            mean = s["mean_" + name]
            means[name].append(None if mean is None else round(mean, 3))

    doc = {
        "id": key, "updated": now.strftime("%Y-%m-%dT%H:%MZ"), "t_start": centre.start, "t_end": centre.end,
        "ir": {
            "source": f"GOES ABI channel {config.ABI_CHANNEL} ({config.ABI_PRODUCT}), brightness temperature",
            "t": [f["t"] for f in frames], "sat": [f["sat"] for f in frames], "valid": [f["valid"] for f in frames],
            "half_km": config.IR_HALF_KM, "res_km": config.IR_RES_KM, "px": px, "cols": cols, "per_sheet": per,
            "sheets": sheets, "tb_max": config.IR_TB_MAX, "tb_step": config.IR_TB_STEP,
        },
        "ccf": {"radius_km": config.RADIUS_KM, "thresholds": list(config.CCF_THRESHOLDS_K),
                "frac": {str(k): [f["ccf"][j] if f["ccf"] is not None else None for f in frames]
                         for j, k in enumerate(config.CCF_THRESHOLDS_K)}},
        "vort": {
            "source": "ERA5 relative vorticity (NCAR, s3://nsf-ncar-era5), layer means",
            "t0": int(hours[0].timestamp()), "dt": 3600, "n": len(hours), "have": have,
            "half_km": config.VORT_HALF_KM, "res_km": config.VORT_RES_KM, "px": n, "unit": config.VORT_UNIT,
            "layers": {"low": list(config.VORT_LOW_HPA), "mid": list(config.VORT_MID_HPA)}, "levels": levels,
            "radius_km": config.RADIUS_KM, "mean_low": means["low"], "mean_mid": means["mid"],     # 1e-5 s-1
            "low": _b64(grids["low"]) if any(have) else "", "mid": _b64(grids["mid"]) if any(have) else "",
        },
    }
    path = os.path.join(config.ENV_DIR, key + ".json")
    with open(path + ".tmp", "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(path + ".tmp", path)
    for stale in os.listdir(config.ENV_DIR):          # sheets left from an earlier, longer version of this record
        if stale.startswith(key + "_") and stale.endswith(".webp") and stale not in sheets:
            os.remove(os.path.join(config.ENV_DIR, stale))
    return doc, ir_bytes + os.path.getsize(path)


def hours_of(centre):
    """Whole hours from the one at or before the start of a record to its end."""
    when = _dt(centre.start).replace(minute=0, second=0, microsecond=0)
    hours = []
    while when.timestamp() <= centre.end:
        hours.append(when)
        when += timedelta(hours=1)
    return hours


def build(product, log=print, now=None):
    """Process one record. Returns its summary for the index (see run_env.py)."""
    began = time.monotonic()
    now = now or datetime.now(UTC)
    key = product["id"]
    centre = Centre(product)

    # ---- IR -----------------------------------------------------------------
    scans = find_scans(centre, log=log)
    url = lambda bucket, k: f"{config.GLM_URL.format(bucket=bucket)}/{k}"
    results = _map("abi", config.ENV_WORKERS, abi.read_frame,
                   [abi.frame_job(url(s["bucket"], s["key"]), *centre(s["t"])) for s in scans])
    # a scan that is missing or mostly bad is tried from the other satellite
    retry = [i for i, (s, r) in enumerate(zip(scans, results)) if s["alt"] and (r["error"] or r["valid"] < 1 - config.IR_MAX_BAD)]
    second = _map("abi", config.ENV_WORKERS, abi.read_frame,
                  [abi.frame_job(url(*scans[i]["alt"]), *centre(scans[i]["t"])) for i in retry])
    cost = results + list(second)
    for i, r in zip(retry, second):
        if not r["error"] and (results[i]["error"] or r["valid"] > results[i]["valid"]):
            results[i], scans[i] = r, dict(scans[i], bucket=scans[i]["alt"][0])
    frames = [dict(r, t=s["t"], sat=abi.SAT_CODE.get(s["bucket"], 0)) for s, r in zip(scans, results) if not r["error"]]
    failed = [r["error"] for r in results if r["error"]]

    # ---- vorticity ----------------------------------------------------------
    hours = hours_of(centre)
    reads = _map("era5", config.ERA5_WORKERS, era5.read_day, era5.jobs_for(hours))
    samples = [era5.sample(when, *centre(when.timestamp())) for when in hours]
    read_errors = [r["error"] for r in reads if r["error"]]
    levels = {"low": next((r["levels_low"] for r in reads if r.get("levels_low")), None),
              "mid": next((r["levels_mid"] for r in reads if r.get("levels_mid")), None)}

    doc, size = write(key, centre, frames, hours, samples, now, levels)
    have = doc["vort"]["have"]
    log(f"  IR: {len(frames)} of {len(scans)} scans read ({sum(r['requests'] for r in cost)} requests, "
        f"{sum(r['bytes'] for r in cost) / 1e6:.0f} MB), {len(doc['ir']['sheets'])} sheets")
    log(f"  ERA5: {sum(have)} of {len(hours)} hours ({sum(r['bytes'] for r in reads) / 1e6:.0f} MB read"
        + (f"; not published yet for {sum(r['missing'] for r in reads)} days" if any(r["missing"] for r in reads) else "")
        + (f"; errors: {read_errors[0]}" if read_errors else "") + ")")
    problems = sorted(set(failed + read_errors))
    return {
        "frames": len(frames), "scans": len(scans), "sheets": len(doc["ir"]["sheets"]),
        "ccf_frames": int(sum(f["ccf"] is not None for f in frames)),
        "hours": len(hours), "vort_hours": int(sum(have)), "era5_pending": bool(any(r["missing"] for r in reads)),
        "sats": sorted({f["sat"] for f in frames}),
        "kb": round(size / 1e3), "seconds": round(time.monotonic() - began),
        "abi_requests": int(sum(r["requests"] for r in cost)), "abi_mb": round(sum(r["bytes"] for r in cost) / 1e6, 1),
        "abi_chunks": next((r.get("chunks") for r in results if r.get("chunks")), None),
        "era5_mb": round(sum(r["bytes"] for r in reads) / 1e6), "era5_seconds": round(sum(r["seconds"] for r in reads)),
        "era5_chunks": next((r.get("chunks") for r in reads if r.get("chunks")), None),
        # an error means nothing usable came out and the case should be tried again; warnings are partial losses
        "error": (problems[0] if problems else "no channel 13 scans found")[:300] if not frames else "",
        "warnings": "; ".join(problems)[:300], "updated": doc["updated"],
    }

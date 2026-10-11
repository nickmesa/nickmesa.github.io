"""tobac lightning features on a storm-following grid.

Same steps and settings as tobac_single_day.py and tobac_merge_pregenesis.py:
feature detection on gridded group count, the "ghost pixel" fix, watershed
segmentation, per-feature sums of group count / area / energy, then linking.

The only structural change is the grid. The paper used one fixed lat/lon box
per storm; here each 5-minute grid is a box centred on the disturbance, cut
from the same global 0.05 deg lattice. Feature positions are converted back to
lattice coordinates before linking, so tracking sees the same geometry.
"""
import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .geo import lattice_latlon

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tobac  # noqa: E402
from tobac.utils import get_statistics_from_mask  # noqa: E402

logging.getLogger("trackpy").setLevel(logging.WARNING)
pd.options.mode.chained_assignment = None

COLUMNS = ["frame", "feature", "gi", "gj", "ncells", "group_count_sum", "area_sum", "energy_sum"]


def _chunk_features(times, count, energy_j, area_m2):
    """Features with statistics for one block of (time, y, x) grids."""
    nt, ny, nx = count.shape
    coords = {"time": pd.DatetimeIndex(times), "y": np.arange(ny), "x": np.arange(nx)}
    group_count = xr.DataArray(count.copy(), dims=("time", "y", "x"), coords=coords, name="group_count")
    # femtojoules and km2, as in tobac_single_day.py
    total_energy = xr.DataArray(energy_j * 1e15, dims=("time", "y", "x"), coords=coords, name="total_energy")
    total_area = xr.DataArray(area_m2 / 1e6, dims=("time", "y", "x"), coords=coords, name="total_group_area")

    features = tobac.feature_detection_multithreshold(group_count, config.GRID_SPACING_M, **config.FEATURE_KW)
    if features is None or len(features) == 0:
        return None

    # Ghost pixels: a weighted centroid can land on an empty pixel, which makes
    # the watershed fail for that feature. Seed that pixel with a 1 for the
    # segmentation and take it out again before any statistics.
    fy = np.clip(np.rint(features["hdim_1"].to_numpy()).astype(int), 0, ny - 1)
    fx = np.clip(np.rint(features["hdim_2"].to_numpy()).astype(int), 0, nx - 1)
    ft = features["frame"].to_numpy().astype(int)
    data = group_count.values
    ghost = data[ft, fy, fx] == 0
    data[ft[ghost], fy[ghost], fx[ghost]] = 1

    mask, masked = tobac.segmentation_2D(features, group_count, config.GRID_SPACING_M, **config.SEGMENT_KW)

    data[ft[ghost], fy[ghost], fx[ghost]] = 0
    if "ncells" not in masked:      # tobac leaves the column out when nothing could be segmented
        masked["ncells"] = 0

    stats = get_statistics_from_mask(masked, mask, group_count, statistic={"group_count_sum": np.sum})
    stats = get_statistics_from_mask(stats, mask, total_area, statistic={"area_sum": np.sum})
    stats = get_statistics_from_mask(stats, mask, total_energy, statistic={"energy_sum": np.sum})
    return stats


def detect(times, ci, cj, cache, half, log=print):
    """Run detection, segmentation and statistics over all steps, a day at a time.

    times: epoch seconds per step; ci, cj: lattice index of the centre per step.
    Returns a DataFrame with lattice positions (gi, gj) and per-feature sums.
    """
    blocks = []
    next_feature = 1
    stamps = pd.to_datetime(np.asarray(times), unit="s")
    for lo in range(0, len(times), config.CHUNK_STEPS):
        hi = min(lo + config.CHUNK_STEPS, len(times))
        count, energy, area = cache.dense(times[lo:hi], ci[lo:hi], cj[lo:hi], half)
        if not count.any():
            continue
        stats = _chunk_features(stamps[lo:hi], count, energy, area)
        if stats is None:
            continue
        frame = stats["frame"].to_numpy().astype(int) + lo
        out = pd.DataFrame({
            "frame": frame,
            "feature": np.arange(next_feature, next_feature + len(stats)),
            "gi": stats["hdim_1"].to_numpy() + (ci[frame] - half),
            "gj": stats["hdim_2"].to_numpy() + (cj[frame] - half),
            "ncells": stats["ncells"].to_numpy(),
            "group_count_sum": pd.to_numeric(stats["group_count_sum"], errors="coerce").to_numpy(),
            "area_sum": pd.to_numeric(stats["area_sum"], errors="coerce").to_numpy(),
            "energy_sum": pd.to_numeric(stats["energy_sum"], errors="coerce").to_numpy(),
        })
        next_feature += len(stats)
        blocks.append(out)
    if not blocks:
        return pd.DataFrame(columns=COLUMNS + ["time"])
    feats = pd.concat(blocks, ignore_index=True)
    feats["time"] = stamps[feats["frame"].to_numpy()]
    return feats


def link(feats):
    """Link features into cells with trackpy and add each cell's lifetime (minutes)."""
    if len(feats) == 0:
        return feats.assign(cell=pd.Series(dtype=int), lifetime_min=pd.Series(dtype=float))
    frame = feats.rename(columns={"gi": "hdim_1", "gj": "hdim_2"}).copy()
    frame["idx"] = 1
    frame["num"] = 1
    frame["threshold_value"] = config.FEATURE_KW["threshold"][0]
    frame["timestr"] = frame["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    dt = config.TIME_RES_MIN * 60
    tracked = tobac.linking_trackpy(frame, None, dt=dt, dxy=config.GRID_SPACING_M, **config.LINK_KW)
    cells = tracked.set_index("feature")["cell"]
    out = feats.sort_values("feature").reset_index(drop=True)
    out["cell"] = out["feature"].map(cells).fillna(-1).astype(int)
    span = out.groupby("cell")["frame"].agg(["min", "max"])
    life = (span["max"] - span["min"] + 1) * config.TIME_RES_MIN
    out["lifetime_min"] = out["cell"].map(life).astype(float)
    out.loc[out["cell"] < 0, "lifetime_min"] = config.TIME_RES_MIN
    return out


def add_derived(feats):
    """Mean group area (km2) and mean group energy (fJ) per feature, and lat/lon."""
    lat, lon = lattice_latlon(feats["gi"].to_numpy(dtype=float), feats["gj"].to_numpy(dtype=float), config.GRID_RES_DEG)
    feats["lat"], feats["lon"] = lat, lon
    n = feats["group_count_sum"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["mean_group_area"] = np.where(n > 0, feats["area_sum"].to_numpy(dtype=float) / n, np.nan)
        feats["mean_group_energy"] = np.where(n > 0, feats["energy_sum"].to_numpy(dtype=float) / n, np.nan)
    return feats

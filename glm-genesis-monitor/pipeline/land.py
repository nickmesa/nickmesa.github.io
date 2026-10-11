"""Distance to land, used to flag lightning cells influenced by coasts and islands."""
import numpy as np

from . import config

_GRID = None


def _load():
    global _GRID
    if _GRID is None:
        with np.load(config.LAND_FILE) as z:
            _GRID = (z["dist"], float(z["lat0"]), float(z["lon0"]), float(z["res"]))     # kept as uint8 km
    return _GRID


def distance_km(lat, lon):
    """Bilinear distance to the nearest coast in km (0 on land, NaN off the grid)."""
    dist, lat0, lon0, res = _load()
    y = (np.asarray(lat, dtype=float) - lat0) / res
    x = (np.asarray(lon, dtype=float) - lon0) / res
    ny, nx = dist.shape
    inside = (y >= 0) & (y <= ny - 1) & (x >= 0) & (x <= nx - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, ny - 2)
    x0 = np.clip(np.floor(x).astype(int), 0, nx - 2)
    fy, fx = np.clip(y - y0, 0, 1), np.clip(x - x0, 0, 1)
    f = np.float32
    out = (dist[y0, x0].astype(f) * (1 - fy) * (1 - fx) + dist[y0 + 1, x0].astype(f) * fy * (1 - fx)
           + dist[y0, x0 + 1].astype(f) * (1 - fy) * fx + dist[y0 + 1, x0 + 1].astype(f) * fy * fx)
    return np.where(inside, out, np.nan)

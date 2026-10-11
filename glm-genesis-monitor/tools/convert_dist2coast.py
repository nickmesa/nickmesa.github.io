"""Use the paper's distance-to-coast file for the land rule.

    python tools/convert_dist2coast.py /path/to/Misc_files/dist2coast_1deg.nc

Reads the 0.01 deg `dist` field from the Dryad package (the file
tobac_merge_pregenesis.py interpolates), keeps the Atlantic / East Pacific
part, and writes pipeline/data/dist2coast.npz. When that file exists the
pipeline uses it instead of the Natural Earth stand-in.

Values are stored as whole km from 0 to 250. Negative and missing values
become 0, so they count as land, which is how the paper code treats them
(cells with dist <= 25 km or NaN are removed).
"""
import os
import sys

import numpy as np
import xarray as xr

LAT0, LAT1, LON0, LON1, CAP_KM = -10.0, 65.0, -180.0, 20.0, 250


def main(path, out=None):
    out = out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline", "data", "dist2coast.npz")
    ds = xr.open_dataset(path)
    da = ds["dist"]
    lat_name = next(n for n in da.dims if n.lower().startswith("lat"))
    lon_name = next(n for n in da.dims if n.lower().startswith("lon"))
    if float(da[lon_name].max()) > 180.0:                       # 0..360 -> -180..180
        da = da.assign_coords({lon_name: ((da[lon_name] + 180.0) % 360.0) - 180.0})
    da = da.sortby(lat_name).sortby(lon_name).transpose(lat_name, lon_name)
    da = da.sel({lat_name: slice(LAT0, LAT1), lon_name: slice(LON0, LON1)})
    lat, lon = da[lat_name].values.astype(float), da[lon_name].values.astype(float)
    res = float(np.round(np.median(np.diff(lat)), 6))
    if not (np.allclose(np.diff(lat), res, atol=res * 1e-3) and np.allclose(np.diff(lon), res, atol=res * 1e-3)):
        raise SystemExit("the dist grid is not regular in latitude and longitude; cannot convert")
    dist = np.empty(da.shape, np.uint8)
    for i in range(0, da.shape[0], 500):                        # in slabs, to keep memory low
        block = da.isel({lat_name: slice(i, i + 500)}).values.astype(np.float32)
        dist[i:i + 500] = np.clip(np.rint(np.nan_to_num(block, nan=0.0)), 0, CAP_KM).astype(np.uint8)
    np.savez_compressed(out, dist=dist, lat0=float(lat[0]), lon0=float(lon[0]), res=res)
    print(f"wrote {out}: {dist.shape[0]} x {dist.shape[1]} at {res} deg, "
          f"{os.path.getsize(out) / 1e6:.1f} MB, {100 * (dist == 0).mean():.1f}% land or missing")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(*sys.argv[1:3])

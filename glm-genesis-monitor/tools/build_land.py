"""Build the distance-to-land grid and the display coastline from Natural Earth.

Run once (the outputs are committed):

    cd glm-genesis-monitor && python tools/build_land.py land-10m.geojson land-50m.geojson

land-*.geojson come from the `world-atlas` npm package (Natural Earth 1:10m
and 1:50m land polygons, public domain) converted with topojson-client.

Outputs
    pipeline/data/dist2land_0p05.npz   distance to the nearest coast in km
                                       (0 on land, capped at 250) on a
                                       0.05 deg grid
    ../lightning/coast.json            1:50m land rings for the page

The paper used a dist2coast file from the GLM climatology package. This grid
is a stand-in built from Natural Earth; to use the original, write it to the
same npz layout (dist, lat0, lon0, res).
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

LAT0, LAT1, LON0, LON1, RES = -10.0, 65.0, -180.0, 20.0, 0.05
SUPER = 5            # rasterise land at RES / SUPER, then sample cell centres
CAP_KM = 250
R_KM = 6372.8


def rings(path):
    gj = json.load(open(path))
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            yield poly      # [exterior, hole, hole, ...]


def xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def main(land10, land50):
    ny = int(round((LAT1 - LAT0) / RES)) + 1
    nx = int(round((LON1 - LON0) / RES)) + 1
    fine = RES / SUPER
    # pixel (row r, col c) of the fine image is centred on lat0 + r*fine, lon0 + c*fine
    h, w = (ny - 1) * SUPER + 1, (nx - 1) * SUPER + 1
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    coast = []
    pad = 3.0
    for poly in rings(land10):
        ext = np.asarray(poly[0], dtype=float)
        if ext[:, 0].max() < LON0 - pad or ext[:, 0].min() > LON1 + pad or ext[:, 1].max() < LAT0 - pad or ext[:, 1].min() > LAT1 + pad:
            continue
        for k, ring in enumerate(poly):
            pts = np.asarray(ring, dtype=float)
            px = [((x - LON0) / fine, (y - LAT0) / fine) for x, y in pts]
            draw.polygon(px, fill=1 if k == 0 else 0)
            # densify the ring so coast points are at most ~1 km apart
            seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
            n = np.maximum(1, np.ceil(seg / 0.01).astype(int))
            for (a, b, m) in zip(pts[:-1], pts[1:], n):
                f = np.linspace(0, 1, m, endpoint=False)[:, None]
                coast.append(a + (b - a) * f)
    coast = np.vstack(coast)
    coast = coast[(coast[:, 0] > LON0 - pad) & (coast[:, 0] < LON1 + pad) & (coast[:, 1] > LAT0 - pad) & (coast[:, 1] < LAT1 + pad)]
    land = np.asarray(img, dtype=bool)[::SUPER, ::SUPER]
    assert land.shape == (ny, nx), land.shape

    lat = LAT0 + np.arange(ny) * RES
    lon = LON0 + np.arange(nx) * RES
    tree = cKDTree(xyz(coast[:, 1], coast[:, 0]))
    dist = np.zeros((ny, nx), np.uint8)
    for i in range(ny):
        chord, _ = tree.query(xyz(np.full(nx, lat[i]), lon), distance_upper_bound=2 * np.sin(CAP_KM / R_KM / 2) * 1.01)
        km = 2 * R_KM * np.arcsin(np.clip(chord / 2, 0, 1))
        km[~np.isfinite(km)] = CAP_KM
        dist[i] = np.clip(np.rint(km), 0, CAP_KM).astype(np.uint8)
    dist[land] = 0
    np.savez_compressed("pipeline/data/dist2land_0p05.npz", dist=dist, lat0=LAT0, lon0=LON0, res=RES)
    print("dist grid", dist.shape, "land fraction", land.mean().round(3))

    # display coastline: 1:50m rings that touch the domain, 2 decimals
    out = []
    for poly in rings(land50):
        for ring in poly:
            pts = np.asarray(ring, dtype=float)
            if pts[:, 0].max() < LON0 or pts[:, 0].min() > LON1 or pts[:, 1].max() < LAT0 or pts[:, 1].min() > LAT1:
                continue
            flat = np.round(pts, 2)
            keep = np.ones(len(flat), bool)
            keep[1:] = np.any(flat[1:] != flat[:-1], axis=1)
            flat = flat[keep]
            if len(flat) >= 4:
                out.append([float(v) for v in flat.ravel()])
    json.dump({"source": "Natural Earth 1:50m land", "rings": out}, open("../lightning/coast.json", "w"), separators=(",", ":"))
    print("coast rings", len(out), "points", sum(len(r) for r in out) // 2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

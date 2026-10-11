"""Write made-up example products to lightning/demo/ so the page can be tried without data.

    python tools/make_demo.py

Nothing produced here is an observation. Tracks and lightning are invented,
then passed through the real analysis code (tobac features, linking, land flag,
200 km filter) so the files have exactly the layout the page reads. The page
only shows them when opened with ?demo=1 and labels them as synthetic.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import atcf, config, glm, process  # noqa: E402
from pipeline.geo import lattice_index  # noqa: E402
from tests.fake_noaa import bdeck_text  # noqa: E402

UTC = timezone.utc
RES = config.GRID_RES_DEG
NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)


def make_track(basin, number, name, start, hours, lat0, lon0, dlat, dlon, kinds, vmax):
    """6-hourly fixes: straight-line motion in deg/h with given types and winds."""
    fixes = []
    for k in range(hours // 6 + 1):
        frac = k / max(hours // 6, 1)
        kind = kinds[min(int(frac * len(kinds)), len(kinds) - 1)]
        v = int(round((vmax[0] + (vmax[1] - vmax[0]) * frac) / 5) * 5)
        wob = 0.25 * np.sin(k / 2.3)
        fixes.append((start + timedelta(hours=6 * k), lat0 + dlat * 6 * k + wob, lon0 + dlon * 6 * k, v, 1012 - v // 4, kind))
    sid = f"{basin}{number}{start.year}"
    return atcf.parse_bdeck(bdeck_text(basin, number, fixes, name=name), sid)


def fill_lightning(cache, steps, rng, activity, trend, sat):
    """Invent convective cells near the centre and write their gridded groups to the cache."""
    n_steps = len(steps)
    hours = (steps.t - steps.t[0]) / 3600.0
    local = (steps.t / 3600.0 + steps.lon / 15.0) % 24
    diurnal = 1.0 + 0.6 * np.cos((local - 5.0) / 24 * 2 * np.pi)            # overnight maximum over the ocean
    envelope = activity * diurnal * trend(hours / max(hours[-1], 1))
    # episodic bursts
    burst = np.zeros(n_steps)
    for _ in range(max(2, int(hours[-1] / 9))):
        c, w = rng.uniform(0, hours[-1]), rng.uniform(1.0, 4.5)
        burst += rng.uniform(0.5, 2.5) * np.exp(-0.5 * ((hours - c) / w) ** 2)
    rate = envelope * (0.08 + burst)                                        # new cells per step
    parts = {k: [] for k in range(n_steps)}
    shear_az = rng.uniform(0, 2 * np.pi)
    for k0 in range(n_steps):
        for _ in range(rng.poisson(rate[k0])):
            broad = rng.random() < 0.22                                      # second population: large, energetic groups
            life = int(np.clip(rng.geometric(0.09 if broad else 0.16), 1, 60))
            r = rng.rayleigh(95.0)
            az = rng.vonmises(shear_az, 0.9)
            y0 = steps.lat[k0] + r * np.cos(az) / 111.2
            x0 = steps.lon[k0] + r * np.sin(az) / (111.2 * np.cos(np.radians(steps.lat[k0])))
            vy, vx = rng.normal(0, 0.012, 2)                                 # deg per step
            size = rng.integers(2, 9) if broad else rng.integers(1, 5)
            lam = rng.lognormal(np.log(6 if broad else 14), 0.8)
            a_mean = rng.lognormal(np.log(760 if broad else 165), 0.38 if broad else 0.30) * 1e6
            e_mean = rng.lognormal(np.log(34 if broad else 4.2), 0.55 if broad else 0.45) * 1e-15
            oy, ox = rng.integers(-1, 2, size), rng.integers(-1, 2, size)
            oy, ox = np.cumsum(oy) - oy[0], np.cumsum(ox) - ox[0]            # a short random walk of touching pixels
            for s in range(life):
                k = k0 + s
                if k >= n_steps:
                    break
                grow = np.sin(np.pi * (s + 0.5) / life) ** 0.7
                gi, gj = lattice_index(y0 + vy * s, x0 + vx * s, RES)
                n = rng.poisson(lam * grow * rng.uniform(0.5, 1.5, size))
                keep = n > 0
                if not keep.any():
                    continue
                n = n[keep].astype(float)
                parts[k].append((gi + oy[keep], gj + ox[keep], n, n * e_mean * rng.uniform(0.8, 1.25, n.size),
                                 n * a_mean * rng.uniform(0.85, 1.2, n.size), n))
    for k in range(n_steps):
        cache.put(steps.t[k], config.EXPECTED_FILES_PER_STEP, steps.ci[k], steps.cj[k], sat, parts[k])


def island_convection(cache, steps, rng, lat, lon):
    """Afternoon thunderstorms that sit on an island, to show the land flag."""
    gi, gj = lattice_index(lat, lon, RES)
    for k in range(len(steps)):
        local = (steps.t[k] / 3600.0 + lon / 15.0) % 24
        if 13 <= local <= 18 and rng.random() < 0.7:
            n = rng.poisson(18, 3).astype(float) + 1
            off = rng.integers(-1, 2)
            part = (np.array([gi, gi, gi + 1]) + off, np.array([gj, gj + 1, gj]), n, n * 6e-15, n * 2.4e8, n)
            old = cache.steps.get(int(steps.t[k]))
            cache._new.append(dict(t=np.full(3, int(steps.t[k]), np.int64), gi=part[0].astype(np.int32),
                                   gj=part[1].astype(np.int32), n=n.astype(np.int32), e=part[3], a=part[4],
                                   av=n.astype(np.int32)))
            assert old is not None


def build(key, track, start, end, status, outcome, genesis, rng, activity, trend, sat=19, extrap=0.0, island=None, note=""):
    steps = process.Steps(track, start, end, extrap_h=extrap)
    cache = glm.CellCache(None)
    fill_lightning(cache, steps, rng, activity, trend, sat)
    if island:
        island_convection(cache, steps, rng, *island)
    feats, info = process.analyse(steps, cache, log=lambda *_: None)
    product = process.build_product(key, track, steps, cache, feats, info, status, outcome, genesis, NOW, note=note)
    product["demo"] = True
    return product


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(os.path.dirname(here), "lightning", "demo")
    config.DATA_DIR = out
    rng = np.random.default_rng(20260914)
    index = {"updated": None, "storms": [], "demo": True}
    data_end = NOW - timedelta(minutes=config.LATENCY_MIN)

    # 1. active Atlantic invest, 81 h of track, convection picking up
    t = make_track("al", "94", "INVEST", datetime(2026, 9, 11, 6, tzinfo=UTC), 78, 9.8, -27.5, 0.022, -0.21,
                   ["WV", "DB", "DB", "LO"], (20, 30))
    products = [build("al942026_0911", t, t.first, data_end, "active", "invest", None, rng, 0.55,
                      lambda f: 0.35 + 1.3 * f ** 2, extrap=config.MAX_EXTRAP_H)]

    # 2. active East Pacific invest seen by GOES-West, weak and intermittent
    t = make_track("ep", "91", "INVEST", datetime(2026, 9, 12, 18, tzinfo=UTC), 42, 12.6, -108.5, 0.012, -0.17,
                   ["DB", "LO"], (20, 25))
    products.append(build("ep912026_0912", t, t.first, data_end, "active", "invest", None, rng, 0.22,
                          lambda f: 1.0 - 0.4 * f, sat=18, extrap=config.MAX_EXTRAP_H))

    # 3. archived developer: 72 h before genesis
    t = make_track("al", "05", "ELARA", datetime(2026, 8, 18, 0, tzinfo=UTC), 120, 11.0, -34.0, 0.03, -0.24,
                   ["WV", "DB", "DB", "LO", "TD", "TS", "TS", "TS"], (20, 55))
    start, end, genesis = process.window_for(t)
    products.append(build("al052026", t, start, end, "ended", "developed", genesis, rng, 0.6,
                          lambda f: 0.25 + 1.6 * f ** 1.5, note="Tracked as Invest 97L before it was numbered."))

    # 4. archived non-developer passing the Windward Islands (land-flagged cells)
    t = make_track("al", "93", "INVEST", datetime(2026, 7, 28, 12, tzinfo=UTC), 84, 12.2, -52.0, 0.008, -0.15,
                   ["WV", "DB", "DB", "WV"], (25, 20))
    products.append(build("al932026_0728", t, t.first, t.last, "ended", "nondeveloper", None, rng, 0.4,
                          lambda f: 1.1 - 0.9 * f, island=(13.19, -59.54)))

    for p in products:
        entry = process.publish(p, index)
        f = p["features"]
        size = os.path.getsize(os.path.join(out, "storms", p["id"] + ".json")) / 1e6
        print(f"{p['id']:16s} {p['label']:18s} steps {p['steps']['n']:5d}  features {len(f['k']):6d} "
              f"(sea {entry['n_features']})  groups {entry['n_groups']:8d}  {size:.2f} MB")
    process.save_index(index, NOW)


if __name__ == "__main__":
    main()

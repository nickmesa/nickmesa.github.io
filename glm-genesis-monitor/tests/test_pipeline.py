"""End-to-end and unit tests. Run with:  python -m pytest -q

The end-to-end tests serve made-up GLM files and b-decks from a local server
(tests/fake_noaa.py) and run the real-time and archive entry points against it.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from scipy import stats

from pipeline import atcf, collect, config, glm, process, run_archive, run_realtime, run_season
from pipeline.geo import haversine, lattice_index
from tests.fake_noaa import FakeNOAA, bdeck_text, lcfa_bytes, quantize_area, quantize_energy

UTC = timezone.utc
FJ = 1e-15
KM2 = 1e6


def dt(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


# ------------------------------------------------------------------ units ---
def test_parse_bdeck_layout():
    text = (
        "AL, 09, 2021082606,   , BEST,   0, 148N,  765W,  25, 1008, DB,   0,    ,    0,    0,    0,    0, 1010,  150,  60,   0,   0,   L,   0,    ,   0,   0,     INVEST, S,  0,    ,    0,    0,    0,    0, genesis-num, 022, TRANSITIONED, alB92021 to al092021, \n"
        "AL, 09, 2021082612,   , BEST,   0, 165N,  788W,  30, 1006, TD,   0,    ,    0,    0,    0,    0, 1010,  150,  60,   0,   0,   L,   0,    ,   0,   0,       NINE, S,\n"
        "AL, 09, 2021082618,   , BEST,   0, 179N,  798W,  35, 1004, TS,  34, NEQ,   60,    0,    0,    0, 1010,  150,  40,   0,   0,   L,   0,    ,   0,   0,        IDA, M,\n"
        "AL, 09, 2021082618,   , BEST,   0, 179N,  798W,  35, 1004, TS,  50, NEQ,    0,    0,    0,    0, 1010,  150,  40,   0,   0,   L,   0,    ,   0,   0,        IDA, M,\n"
        "AL, 09, 2021082916, 55, BEST,   0, 291N,  902W, 130,  931, HU,  34, NEQ,  120,  100,   70,  100, 1008,  180,  10,   0,   0,   L,   0,    ,   0,   0,        IDA, D,\n"
    )
    tr = atcf.parse_bdeck(text, "al092021")
    assert len(tr.times) == 4                      # the two 18Z radius lines collapse to one fix
    assert tr.lat[0] == 14.8 and tr.lon[0] == -76.5
    assert tr.kind == ["DB", "TD", "TS", "HU"]
    assert tr.genesis_time() == dt("2021-08-26T18:00")     # first TS, as in tracks_200km_72hr.py
    assert tr.last == dt("2021-08-29T16:55")               # minutes column
    assert tr.name == "IDA" and not tr.is_invest and tr.transitioned_from == "alb92021"
    assert process.storm_label(tr) == "Ida (AL09)"


def test_centre_interpolation_and_extrapolation():
    fixes = [(dt("2026-10-09T18:00"), 12.0, -39.0, 25, 1010, "DB"), (dt("2026-10-10T00:00"), 12.6, -40.5, 25, 1010, "DB")]
    tr = atcf.parse_bdeck(bdeck_text("al", "92", fixes), "al922026")
    when = np.array(["2026-10-09T21:00", "2026-10-10T03:00", "2026-10-10T13:00", "2026-10-09T17:00"], dtype="datetime64[s]")
    lat, lon, ex = tr.centre(when, extrap_h=12)
    assert lat[0] == pytest.approx(12.3) and lon[0] == pytest.approx(-39.75) and not ex[0]
    assert lat[1] == pytest.approx(12.9) and lon[1] == pytest.approx(-41.25) and ex[1]     # same motion carried forward
    assert np.isnan(lat[2]) and np.isnan(lat[3])                                         # beyond 12 h; before first fix
    assert tr.is_invest and process.storm_label(tr) == "Invest 92L"


def test_satellite_choice():
    assert glm.bucket_for(dt("2024-06-28T00:00"), -45.0) == "noaa-goes16"
    assert glm.bucket_for(dt("2026-10-10T00:00"), -45.0) == "noaa-goes19"
    assert glm.bucket_for(dt("2022-08-01T00:00"), -115.0) == "noaa-goes17"
    assert glm.bucket_for(dt("2026-10-10T00:00"), -115.0) == "noaa-goes18"
    assert glm.bucket_for(dt("2026-10-10T00:00"), -100.0) == "noaa-goes19"   # eastern East Pacific stays on GOES-East


def test_read_groups_unpacks_and_filters():
    e = np.array([4, 8, 300]) * FJ
    a = np.array([100, 300, 5000]) * KM2
    raw = lcfa_bytes(lat=[10.0, 10.1, 10.2], lon=[-40.0, -40.1, -40.2], energy_j=e, area_m2=a, flag=[0, 1, 0])
    lat, lon, energy, area = glm.read_groups(raw)
    assert len(lat) == 2                                         # flagged group dropped
    np.testing.assert_allclose(energy, quantize_energy(e)[[0, 2]], rtol=1e-6)
    np.testing.assert_allclose(area, quantize_area(a)[[0, 2]], rtol=1e-6)
    assert energy[1] > 2.9e-13                                   # unsigned 16-bit packing handled (would be negative otherwise)
    # files before the 2019 unit change stored area in km2
    lat, lon, energy, area_km = glm.read_groups(lcfa_bytes(lat=[10.0], lon=[-40.0], energy_j=e[:1], area_m2=a[:1], area_units="km2"))
    np.testing.assert_allclose(area_km, quantize_area(a[:1]), rtol=1e-6)
    assert len(glm.read_groups(lcfa_bytes(lat=[], lon=[], energy_j=[], area_m2=[]))[0]) == 0


def test_binning_matches_paper_grid():
    """bin_groups gives the same counts and sums as GET_GRID + bin_statistic in Functions.py."""
    rng = np.random.default_rng(1)
    lat_min, lat_max, lon_min, lon_max, res = 8.0, 16.0, -44.0, -36.0, 0.05
    lat = rng.uniform(lat_min + 0.5, lat_max - 0.5, 20000)
    lon = rng.uniform(lon_min + 0.5, lon_max - 0.5, 20000)
    energy = rng.uniform(1, 50, lat.size) * FJ
    area = rng.uniform(60, 2000, lat.size) * KM2
    lat_arr = np.round(np.arange(lat_min, lat_max + res, res), 6)
    lon_arr = np.round(np.arange(lon_min, lon_max + res, res), 6)
    lat_b = np.round(np.append(lat_arr, lat_arr[-1] + res) - res / 2.0, 6)
    lon_b = np.round(np.append(lon_arr, lon_arr[-1] + res) - res / 2.0, 6)
    ref_n = stats.binned_statistic_2d(lon, lat, energy, "count", bins=[lon_b, lat_b]).statistic.T
    ref_e = stats.binned_statistic_2d(lon, lat, energy, "sum", bins=[lon_b, lat_b]).statistic.T
    ref_a = stats.binned_statistic_2d(lon, lat, area, "mean", bins=[lon_b, lat_b]).statistic.T

    ci, cj = lattice_index(12.0, -40.0, res)
    half = 80
    gi, gj, n, e, a, av = glm.bin_groups(lat, lon, energy, area, int(ci), int(cj), half)
    i0, j0 = lattice_index(lat_min, lon_min, res)
    mine_n = np.zeros_like(ref_n)
    mine_e = np.zeros_like(ref_n)
    mine_a = np.full_like(ref_n, np.nan)
    mine_n[gi - i0, gj - j0] = n
    mine_e[gi - i0, gj - j0] = e
    mine_a[gi - i0, gj - j0] = a / av
    assert (mine_n != ref_n).sum() <= 2          # only points lying on a bin edge to float precision may differ
    same = mine_n == ref_n
    np.testing.assert_allclose(mine_e[same], ref_e[same], rtol=1e-9)
    np.testing.assert_allclose(mine_a[same & (ref_n > 0)], ref_a[same & (ref_n > 0)], rtol=1e-9)


# ------------------------------------------------------------- end to end ---
T00 = dt("2026-10-10T00:00")
A_PIX = [(12.50, -40.50, 2, 4, 100), (12.50, -40.45, 1, 8, 300), (12.55, -40.50, 1, 8, 500)]   # lat, lon, groups/file, fJ, km2


def groups_at(start, hour2=False):
    """Made-up groups in the 20 s file starting at `start` (GOES-East)."""
    step = int((start - T00).total_seconds() // 300)
    g = []   # lat, lon, energy fJ, area km2, flag
    jit = 0.012 * np.sin(start.second + start.minute)          # keeps points off the cell centre but inside the cell
    if 0 <= step <= 5:
        for la, lo, n, e, a in A_PIX:                           # A: three touching pixels, 56 km from the centre
            g += [(la + jit, lo - jit, e, a, 0)] * n
        g += [(14.6 + jit, -40.5, 5, 200, 0)] * 2               # B: 290 km north, outside the 200 km circle
        g += [(13.20, -59.30 + jit, 5, 200, 0)]                 # L1: 15 km off Barbados (second invest)
        g += [(13.50, -57.50 + jit, 5, 200, 0)]                 # L2: open water east of Barbados
    if 0 <= step < 12:
        g += [(12.0, -40.6 + jit, 50, 900, 1)] * 3              # C: bad-quality groups, must be ignored
    if 12 <= step < 20:
        g += [(11.5, -41.0 + 0.05 * (step - 12) + jit, 3, 150, 0)] * 3     # D: one pixel moving a pixel east per step
    if step in (12, 13):
        for la, lo in ((11.0, -40.0), (11.0, -39.9), (11.05, -39.95)):     # E: V shape with an empty centroid pixel
            g += [(la + jit, lo + jit, 6, 250, 0)]
    if 24 <= step <= 26:
        g += [(12.0, -41.2 + jit, 7, 120, 0)] * 2               # F: appears in the third hour
    return g


def put_hours(noaa, bucket, first, hours, maker, skip=()):
    t = first
    while t < first + timedelta(hours=hours):
        if t not in skip:
            g = maker(t)
            noaa.put_glm(bucket, t, lat=[r[0] for r in g], lon=[r[1] for r in g], energy_j=[r[2] * FJ for r in g],
                         area_m2=[r[3] * KM2 for r in g], flag=[r[4] for r in g])
        t += timedelta(seconds=20)


def fixes_west(first, n, lat=12.0, lon0=-39.0, kinds=None):
    """6-hourly fixes moving west at 0.25 deg/h."""
    out = []
    for k in range(n):
        kind = kinds[k] if kinds else "DB"
        out.append((first + timedelta(hours=6 * k), lat, lon0 - 1.5 * k, 35 if kind == "TS" else 25, 1009, kind))
    return out


@pytest.fixture()
def world(tmp_path, monkeypatch):
    noaa = FakeNOAA()
    base = noaa.start()
    monkeypatch.setattr(config, "ATCF_URL", base + "/atcf")
    monkeypatch.setattr(config, "GLM_URL", base + "/{bucket}")
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    yield noaa, tmp_path
    noaa.stop()


def load(tmp_path, key=None):
    path = tmp_path / "data" / ("index.json" if key is None else f"storms/{key}.json")
    return json.loads(path.read_text())


def feature_rows(prod, **where):
    f = prod["features"]
    rows = [dict(zip(f, vals)) for vals in zip(*f.values())]
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def test_realtime_lifecycle(world):
    noaa, tmp = world
    first_fix = dt("2026-10-09T18:00")
    noaa.btk["bal922026.dat"] = bdeck_text("al", "92", fixes_west(first_fix, 2))
    noaa.btk["bal932026.dat"] = bdeck_text("al", "93", [(first_fix, 13.5, -58.5, 20, 1011, "DB"), (T00, 13.5, -58.5, 20, 1011, "DB")])
    noaa.btk["bep912026.dat"] = bdeck_text("ep", "91", [(first_fix, 12.0, -115.0, 20, 1010, "LO"), (T00, 12.0, -115.0, 20, 1010, "LO")])
    noaa.btk["bcp012026.dat"] = bdeck_text("cp", "01", [(T00, 15.0, -150.0, 30, 1005, "TD")])    # central Pacific: ignored
    late = dt("2026-10-10T01:55:40")
    put_hours(noaa, "noaa-goes19", T00, 2, groups_at, skip={late})
    west = lambda t: [(12.3, -115.2, 9, 400, 0)] * 2 if t < T00 + timedelta(minutes=10) else []
    put_hours(noaa, "noaa-goes18", T00, 1, west)

    # ---- run 1: 02:10Z -----------------------------------------------------
    run_realtime.main(["--now", "2026-10-10T02:10"])
    index = load(tmp)
    ids = {e["id"]: e for e in index["storms"]}
    assert set(ids) == {"al922026_1009", "al932026_1009", "ep912026_1009"}
    assert all(e["status"] == "active" and e["outcome"] == "invest" for e in ids.values())
    assert len(noaa.gets) == 359 + 180                       # two GOES-East hours (one file late) + one GOES-West hour

    p = load(tmp, "al922026_1009")
    s = p["steps"]
    assert s["t0"] == int(first_fix.timestamp()) and s["n"] == 96          # 18:00 to 02:00
    assert s["nfiles"][:72] == [0] * 72 and s["nfiles"][72:95] == [15] * 23 and s["nfiles"][95] == 14
    assert s["extrap_from"] == 73                                          # first step after the 00Z fix
    assert s["clon"][72] == pytest.approx(-40.5) and s["clon"][84] == pytest.approx(-40.75)
    assert s["sat"] == [[0, 19]]

    # A: 6 steps, one feature each, sums follow the paper's per-pixel arithmetic
    a_rows = [r for r in feature_rows(p) if 72 <= r["k"] <= 77 and r["y"] > 40]
    assert len(a_rows) == 6
    e_fj = quantize_energy(np.array([4, 8, 8]) * FJ) / FJ
    a_km = quantize_area(np.array([100, 300, 500]) * KM2) / KM2
    for r in a_rows:
        assert r["n"] == 60 and r["px"] == 3 and r["land"] == 0
        assert r["energy"] == pytest.approx((30 * e_fj[0] + 15 * e_fj[1] + 15 * e_fj[2]) / 60, abs=0.006)
        assert r["area"] == pytest.approx((30 * a_km[0] + 15 * a_km[1] + 15 * a_km[2]) / 60, abs=0.06)
        assert r["lat"] == pytest.approx(12.512, abs=0.01) and r["lon"] == pytest.approx(-40.488, abs=0.01)
        assert r["life"] == 30                                              # one cell lasting six steps
    assert len({r["cell"] for r in a_rows}) == 1
    first = a_rows[0]
    assert first["x"] == pytest.approx(0.0, abs=2.5) and first["y"] == pytest.approx(56.9, abs=1.5)
    # the centre moves west, so the same cloud drifts east in the storm-relative frame
    assert a_rows[-1]["x"] - first["x"] == pytest.approx(0.25 * 25 / 60 * 111.2 * np.cos(np.radians(12.25)), abs=1.5)

    # B (outside 200 km) and C (bad quality) leave no trace; B was still seen by tobac
    assert all(haversine(12.0, -40.5, r["lat"], r["lon"]) < 260 for r in feature_rows(p))
    assert not [r for r in feature_rows(p) if r["energy"] > 40]
    assert p["info"]["features_in_domain"] == len(feature_rows(p)) + 6

    # D: one pixel moving east is one cell lasting 8 steps
    d_rows = [r for r in feature_rows(p) if 84 <= r["k"] < 92 and r["y"] < -40 and r["n"] == 45 and r["px"] == 1]
    assert len(d_rows) == 8 and len({r["cell"] for r in d_rows}) == 1 and d_rows[0]["life"] == 40
    assert d_rows[3]["lon"] - d_rows[2]["lon"] == pytest.approx(0.05, abs=1e-6)

    # E: the centroid falls on an empty pixel; the ghost-pixel step keeps the feature and its three pixels
    # (tobac's pixel count includes the seeded pixel, the group sum does not)
    e_rows = [r for r in feature_rows(p) if r["k"] in (84, 85) and r["px"] == 4]
    assert len(e_rows) == 2 and all(r["n"] == 45 for r in e_rows)
    assert p["info"]["unsegmented"] == 0

    # land flag near Barbados
    q = load(tmp, "al932026_1009")
    near = [r for r in feature_rows(q) if r["lon"] < -59]
    far = [r for r in feature_rows(q) if r["lon"] > -58]
    assert len(near) == 6 and all(r["land"] == 1 for r in near)
    assert len(far) == 6 and all(r["land"] == 0 for r in far)
    assert ids["al932026_1009"]["n_features"] == 6 and ids["al932026_1009"]["n_groups"] == 6 * 15

    # East Pacific invest west of 106W uses GOES-West
    w = load(tmp, "ep912026_1009")
    assert w["steps"]["sat"] == [[0, 18]] and len(feature_rows(w)) == 2 and feature_rows(w)[0]["n"] == 30
    assert w["label"] == "Invest 91E"

    # ---- run 2: 03:10Z, one more hour and the late file has arrived ---------
    put_hours(noaa, "noaa-goes19", T00 + timedelta(hours=2), 1, lambda t: groups_at(t))
    put_hours(noaa, "noaa-goes19", late, 0.005, groups_at)
    put_hours(noaa, "noaa-goes18", T00 + timedelta(hours=1), 2, lambda t: [])
    before = len(noaa.gets)
    run_realtime.main(["--now", "2026-10-10T03:10"])
    new = noaa.gets[before:]
    # GOES-East: hour 02 once (shared by 92L and 93L) + the incomplete 01:55 step again; GOES-West: two empty hours
    assert len(new) == 180 + 15 + 360
    p2 = load(tmp, "al922026_1009")
    assert p2["steps"]["n"] == 108 and p2["steps"]["nfiles"][95] == 15
    f_rows = [r for r in feature_rows(p2) if r["k"] in (96, 97, 98)]
    assert len(f_rows) == 3 and all(r["n"] == 30 for r in f_rows)
    assert feature_rows(p2)[: len(feature_rows(p))] == feature_rows(p) or len(feature_rows(p2)) == len(feature_rows(p)) + 3

    # ---- run 3: 04:10Z, 92L is numbered as AL10 -----------------------------
    del noaa.btk["bal922026.dat"]
    noaa.btk["bal102026.dat"] = bdeck_text("al", "10", fixes_west(first_fix, 2, kinds=["DB", "TD"]), name="TEN")
    put_hours(noaa, "noaa-goes19", T00 + timedelta(hours=3), 1, lambda t: [])
    put_hours(noaa, "noaa-goes18", T00 + timedelta(hours=3), 1, lambda t: [])
    before = len(noaa.gets)
    run_realtime.main(["--now", "2026-10-10T04:10"])
    ids = {e["id"]: e for e in load(tmp)["storms"]}
    assert "al922026_1009" not in ids and not (tmp / "data/storms/al922026_1009.json").exists()
    ten = ids["al102026"]
    assert ten["status"] == "active" and ten["outcome"] == "pre-genesis" and "Invest 92L" in ten["note"]
    assert ten["label"] == "AL10"                       # "TEN" is a placeholder name, not shown
    assert len(noaa.gets) - before == 180 + 180         # only the new hour; earlier steps came from 92L's cache
    p3 = load(tmp, "al102026")
    assert len(feature_rows(p3)) == len(feature_rows(p2)) and p3["steps"]["n"] == 120

    # ---- run 4: 13:10Z, AL10 reaches tropical-storm strength at 12Z ----------
    noaa.btk["bal102026.dat"] = bdeck_text("al", "10", fixes_west(first_fix, 4, kinds=["DB", "TD", "TD", "TS"]), name="JULIA")
    run_realtime.main(["--now", "2026-10-10T13:10"])
    ids = {e["id"]: e for e in load(tmp)["storms"]}
    ten = ids["al102026"]
    assert ten["status"] == "ended" and ten["outcome"] == "developed" and ten["genesis"] == "2026-10-10T12:00Z"
    assert ten["label"] == "Julia (AL10)" and ten["pending_steps"] == 0
    p4 = load(tmp, "al102026")
    assert p4["steps"]["n"] == 216 and p4["steps"]["extrap_from"] == -1        # 18Z to genesis at 12Z, all interpolated
    assert not (tmp / "cache/al102026.npz").exists()
    assert ids["al932026_1009"]["status"] == "active"                          # last fix 13 h old: still monitored
    gets_after_4 = len(noaa.gets)

    # ---- run 5: 19:10Z, 93L and 91E have had no fix for 19 h ------------------
    run_realtime.main(["--now", "2026-10-10T19:10"])
    ids = {e["id"]: e for e in load(tmp)["storms"]}
    assert ids["al932026_1009"]["status"] == "ended" and ids["al932026_1009"]["outcome"] == "nondeveloper"
    assert load(tmp, "al932026_1009")["outcome"] == "nondeveloper"
    assert ids["al102026"]["outcome"] == "developed" and len(noaa.gets) == gets_after_4    # finished storm is not redone
    assert not list((tmp / "cache").glob("*.npz"))


def test_archive_run(world):
    noaa, tmp = world
    gen = dt("2024-06-29T00:00")
    fixes = [(gen - timedelta(hours=6 * (3 - k)), 9.0 + 0.1 * k, -38.0 - 1.5 * k, 35 if k == 3 else 25, 1008, "TS" if k == 3 else "DB")
             for k in range(4)] + [(gen + timedelta(hours=6), 9.5, -45.0, 60, 995, "TS")]
    noaa.archive["2024/bal022024.dat.gz"] = bdeck_text("al", "02", fixes, name="BERYL")
    centre_lon = lambda t: -42.5 + 0.25 * (gen - t).total_seconds() / 3600
    maker = lambda t: [(9.3, centre_lon(t) + 0.5, 3, 180, 0)] * 2          # a cluster riding 55 km east of the centre
    put_hours(noaa, "noaa-goes16", gen - timedelta(hours=1), 1, maker)
    prod = run_archive.run("al022024")
    assert prod["outcome"] == "developed" and prod["genesis"] == "2024-06-29T00:00Z" and prod["label"] == "Beryl (AL02)"
    assert prod["steps"]["n"] == 18 * 12                                    # track starts 18 h before genesis
    assert prod["steps"]["sat"][-1][1] == 16
    rows = feature_rows(prod)
    assert len(rows) == 12 and all(r["n"] == 30 for r in rows)
    assert all(r["x"] == pytest.approx(54.8, abs=3.5) for r in rows)
    entry = load(tmp)["storms"][0]
    assert entry["id"] == "al022024" and entry["status"] == "ended" and sum(entry["spark"]) == 360

    # an archived invest with an explicit window
    noaa.archive["2024/invests/balA62024.dat.gz"] = bdeck_text("al", "A6", fixes[:3])
    prod = run_archive.run("alA62024", start=gen - timedelta(hours=12), end=gen - timedelta(hours=6))
    assert prod["label"] == "Invest 96L" and prod["outcome"] == "nondeveloper" and prod["steps"]["n"] == 72


def test_convert_dist2coast(tmp_path, monkeypatch):
    """The paper's dist file (here a coarse stand-in on a 0-360 grid) converts to the layout land.py reads."""
    import xarray as xr
    from pipeline import land
    from tools import convert_dist2coast
    lat = np.arange(-90, 90.01, 0.5)[::-1]                    # north to south, as many products are stored
    lon = np.arange(0, 360, 0.5)
    lon2, lat2 = np.meshgrid(lon, lat)
    dist = 100.0 + lat2 + ((lon2 + 180) % 360 - 180) / 10.0   # a smooth made-up field
    dist[(lat2 > 40) & (lon2 > 300)] = -30.0                   # "land" block
    dist[(lat2 < -5) & (lon2 > 300)] = np.nan
    src = tmp_path / "dist2coast_1deg.nc"
    xr.Dataset({"dist": (("lat", "lon"), dist)}, coords={"lat": lat, "lon": lon}).to_netcdf(src)
    out = tmp_path / "dist2coast.npz"
    convert_dist2coast.main(str(src), str(out))
    monkeypatch.setattr(config, "LAND_FILE", str(out))
    monkeypatch.setattr(land, "_GRID", None)
    assert float(land.distance_km(12.0, -40.0)) == pytest.approx(100 + 12.0 - 4.0, abs=0.6)
    assert float(land.distance_km(20.25, -75.25)) == pytest.approx(100 + 20.25 - 7.525, abs=0.6)
    assert float(land.distance_km(50.0, -50.0)) == 0.0         # negative values count as land
    assert float(land.distance_km(-8.0, -50.0)) == 0.0         # so do missing values
    assert np.isnan(land.distance_km(70.0, -40.0))             # outside the stored domain
    monkeypatch.setattr(land, "_GRID", None)


def test_archive_joins_precursor_invest(world):
    """NHC's archived best track starts near genesis; the days before come from the invest archive."""
    noaa, tmp = world
    gen = dt("2024-08-10T12:00")
    line = lambda k: (10.0 + 0.1 * k, -40.0 - 1.5 * k)             # the system's path, k = 6-hourly index
    best = [(gen - timedelta(hours=6), *line(10), 30, 1007, "TD"), (gen, *line(11), 35, 1005, "TS"),
            (gen + timedelta(hours=6), *line(12), 40, 1003, "TS")]
    noaa.archive["2024/bal052024.dat.gz"] = bdeck_text("al", "05", best, name="ERNESTO")
    right = [(gen - timedelta(hours=6 * (11 - k)), *line(k), 20, 1010, "DB") for k in range(1, 11)]   # -60 h to -6 h
    far = [(t, la + 8.0, lo, v, p, ty) for t, la, lo, v, p, ty in right]                              # same times, ~890 km north
    early = [(t - timedelta(days=30), la, lo, v, p, ty) for t, la, lo, v, p, ty in right]             # same path, a month before
    noaa.archive["2024/invests/balA12024.dat.gz"] = bdeck_text("al", "A1", early)
    noaa.archive["2024/invests/balB32024.dat.gz"] = bdeck_text("al", "B3", right)
    noaa.archive["2024/invests/balC32024.dat.gz"] = bdeck_text("al", "C3", far)
    noaa.archive["2024/invests/bepA12024.dat.gz"] = bdeck_text("ep", "A1", right)                     # other basin
    prod = run_archive.run("al052024")
    assert prod["steps"]["n"] == 60 * 12                         # 60 h of positions instead of 6
    assert "alB32024" in prod["note"] and "60 h" in prod["note"]
    assert prod["track"]["type"][:3] == ["DB", "DB", "DB"] and prod["genesis"] == "2024-08-10T12:00Z"
    assert prod["label"] == "Ernesto (AL05)" and prod["steps"]["clat"][0] == pytest.approx(10.1)

    # a TRANSITIONED note in the invest's b-deck decides the match even when another invest is closer
    marked = [(t, la + 4.0, lo, v, p, ty) for t, la, lo, v, p, ty in right]
    text = bdeck_text("al", "D7", marked).replace("genesis-num, 040, ", "genesis-num, 040, TRANSITIONED, alD72024 to al052024, ")
    noaa.archive["2024/invests/balD72024.dat.gz"] = text
    prod = run_archive.run("al052024")
    assert "alD72024" in prod["note"] and prod["steps"]["clat"][0] == pytest.approx(14.1)

    # with no usable invest the record is just shorter, and says so
    for k in list(noaa.archive):
        if "/invests/" in k:
            del noaa.archive[k]
    prod = run_archive.run("al052024")
    assert prod["steps"]["n"] == 6 * 12 and "Only 6 h" in prod["note"]


def test_archive_uses_combined_best_track_file(world, monkeypatch):
    """A BEST_b<id>.nc file from the paper's Misc_files takes the place of NHC's b-decks."""
    import xarray as xr
    noaa, tmp = world
    gen = dt("2021-08-26T18:00")
    noaa.archive["2021/bal092021.dat.gz"] = bdeck_text("al", "09", [(gen - timedelta(hours=6), 16.5, -78.9, 30, 1006, "TD"),
                                                                    (gen, 17.4, -79.5, 35, 1004, "TS")], name="IDA")
    times = np.array([np.datetime64("2021-08-22T18:00") + np.timedelta64(6 * k, "h") for k in range(20)]).astype("datetime64[ns]")
    kinds = np.array([b"WV"] * 6 + [b"DB"] * 9 + [b"TD"] + [b"TS"] * 4, dtype="S2")
    tracks = tmp / "tracks"
    tracks.mkdir()
    xr.Dataset({"track_lat": ("initialization_time", 12.0 + 0.3 * np.arange(20)),
                "track_lon": ("initialization_time", 360.0 - 70.0 - 0.6 * np.arange(20)),      # degrees east, 0-360
                "track_type": ("initialization_time", kinds),
                "track_maxwind": ("initialization_time", 20.0 + np.arange(20))},
               coords={"initialization_time": times}).to_netcdf(tracks / "BEST_bal092021.nc")
    monkeypatch.setattr(config, "TRACK_DIR", str(tracks))
    prod = run_archive.run("al092021")
    assert prod["genesis"] == "2021-08-26T18:00Z" and prod["steps"]["n"] == 72 * 12
    assert prod["label"] == "Ida (AL09)" and "combined invest and best-track file" in prod["note"]
    assert prod["steps"]["clon"][0] == pytest.approx(-72.4) and prod["steps"]["clat"][0] == pytest.approx(13.2)


def test_season_backfill(world, monkeypatch):
    """A season is every numbered system plus every invest that did not become one."""
    noaa, tmp = world
    gen = dt("2024-08-10T12:00")
    line = lambda k: (10.0 + 0.1 * k, -40.0 - 1.5 * k)
    six_h = lambda t0, n, lat, lon, kind="DB": [(t0 + timedelta(hours=6 * k), lat, lon - 0.5 * k, 25, 1009, kind) for k in range(n)]
    noaa.archive["2024/bal052024.dat.gz"] = bdeck_text("al", "05", [(gen - timedelta(hours=6), *line(10), 30, 1007, "TD"),
                                                                    (gen, *line(11), 35, 1005, "TS")], name="ERNESTO")
    grew = [(gen - timedelta(hours=6 * (11 - k)), *line(k), 20, 1010, "DB") for k in range(7, 11)]    # -24 h to -6 h
    noaa.archive["2024/invests/balB32024.dat.gz"] = bdeck_text("al", "B3", grew)
    noaa.archive["2024/bal062024.dat.gz"] = bdeck_text("al", "06", six_h(dt("2024-09-01T00:00"), 3, 25.0, -60.0, "TD"), name="SIX")
    noaa.archive["2024/invests/balA12024.dat.gz"] = bdeck_text("al", "A1", six_h(dt("2024-07-01T00:00"), 4, 12.0, -30.0))
    noaa.archive["2024/invests/balE22024.dat.gz"] = bdeck_text("al", "E2", six_h(dt("2024-07-20T00:00"), 2, 12.0, -30.0))
    # an Atlantic invest that crossed into the East Pacific and was numbered there
    gen2 = dt("2024-06-20T00:00")
    noaa.archive["2024/bep012024.dat.gz"] = bdeck_text("ep", "01", [(gen2 - timedelta(hours=6), 12.0, -92.0, 30, 1007, "TD"),
                                                                    (gen2, 12.2, -93.0, 35, 1005, "TS")], name="ALETTA")
    noaa.archive["2024/invests/balF42024.dat.gz"] = bdeck_text("al", "F4", six_h(gen2 - timedelta(hours=24), 4, 11.9, -88.9))
    noaa.archive["2024/aal052024.dat.gz"] = "not a best track"

    now = dt("2026-10-11T12:00")
    cases = {c["id"]: c for c in run_season.run(2024, "al", dry_run=True, now=now)}
    assert sorted(cases) == ["al052024", "al062024", "alA12024", "alB32024", "alE22024", "alF42024"]
    assert cases["al052024"]["outcome"] == "developed" and cases["al052024"]["precursor"] == "alB32024"
    assert cases["al052024"]["hours"] == 24 and cases["al052024"]["label"] == "Ernesto (AL05)"
    assert cases["al062024"]["outcome"] == "nondeveloper" and cases["al062024"]["hours"] == 12
    assert cases["alA12024"]["status"] == "pending" and cases["alA12024"]["label"] == "Invest 91L"
    assert cases["alB32024"]["reason"] == "became al052024" and cases["alF42024"]["reason"] == "became ep012024"
    assert cases["alE22024"]["status"] == "skipped" and "shorter" in cases["alE22024"]["reason"]
    plan = json.loads((tmp / "data/backfill/2024-al.json").read_text())
    assert plan["counts"] == {"pending": 3, "skipped": 3} and not (tmp / "data/storms").exists()

    # one case failing does not stop the others, and the plan records why
    real_run = run_archive.run
    def flaky(sid, **kw):
        if sid == "al062024":
            raise RuntimeError("bucket unreachable")
        return real_run(sid, **kw)
    monkeypatch.setattr(run_archive, "run", flaky)
    cases = {c["id"]: c for c in run_season.run(2024, "al", now=now)}
    assert cases["al062024"]["status"] == "failed" and "bucket unreachable" in cases["al062024"]["reason"]
    assert cases["al052024"]["status"] == "done" and cases["al052024"]["n_steps"] == 24 * 12
    assert sorted(e["id"] for e in load(tmp)["storms"]) == ["al052024", "alA12024"]
    assert "alB32024" in load(tmp, "al052024")["note"]

    # a second run only does what is missing
    ran = []
    monkeypatch.setattr(run_archive, "run", lambda sid, **kw: ran.append(sid) or real_run(sid, **kw))
    cases = {c["id"]: c for c in run_season.run(2024, "al", now=now)}
    assert ran == ["al062024"] and cases["al052024"]["reason"] == "already in the archive"
    assert json.loads((tmp / "data/backfill/2024-al.json").read_text())["counts"] == {"done": 3, "skipped": 3}
    assert load(tmp, "al062024")["outcome"] == "nondeveloper" and load(tmp, "al062024")["genesis"] is None
    assert run_season.run(2024, "al", max_hours=0, redo=True, now=now)[0]["status"] == "pending"     # out of time: nothing started

    # the other basin is processed into its own folder, as a parallel job would, then collected
    monkeypatch.setattr(config, "DATA_DIR", str(tmp / "job" / "data"))
    cases = {c["id"]: c for c in run_season.run(2024, "ep", now=now)}
    assert list(cases) == ["ep012024"] and cases["ep012024"]["precursor"] == "alF42024" and cases["ep012024"]["hours"] == 24
    monkeypatch.setattr(config, "DATA_DIR", str(tmp / "data"))
    assert collect.collect(str(tmp / "job"), now=now, log=lambda m: None) == (1, 1)
    assert sorted(e["id"] for e in load(tmp)["storms"]) == ["al052024", "al062024", "alA12024", "ep012024"]
    assert (tmp / "data/backfill/2024-ep.json").exists() and (tmp / "data/backfill/2024-al.json").exists()


def test_season_in_progress(world):
    """The current season is not in NHC's archive yet: numbered systems come from the real-time folder."""
    noaa, tmp = world
    now = dt("2026-10-11T12:00")
    gen = dt("2026-07-04T00:00")
    noaa.btk["bal012026.dat"] = bdeck_text("al", "01", [(gen - timedelta(hours=6 * k), 15.0, -50.0 + k, 25 if k else 35, 1008, "DB" if k else "TS")
                                                        for k in range(8, -1, -1)], name="ARTHUR")
    noaa.btk["bal102026.dat"] = bdeck_text("al", "10", [(now - timedelta(hours=18), 12.0, -40.0, 30, 1007, "TD"),
                                                        (now - timedelta(hours=12), 12.2, -41.0, 30, 1007, "TD")], name="TEN")
    noaa.btk["bal932026.dat"] = bdeck_text("al", "93", [(now - timedelta(hours=12), 9.0, -30.0, 20, 1010, "DB"),
                                                        (now - timedelta(hours=6), 9.0, -31.0, 20, 1010, "DB")])
    cases = {c["id"]: c for c in run_season.run(2026, "al", dry_run=True, now=now)}
    assert sorted(cases) == ["al012026", "al102026"]                      # live invests belong to the hourly run
    assert cases["al012026"]["status"] == "pending" and cases["al012026"]["hours"] == 48
    assert cases["al102026"]["status"] == "skipped" and "hourly run" in cases["al102026"]["reason"]

"""NHC ATCF b-decks: find disturbances, read their tracks, interpolate centres."""
import gzip
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from . import config

UTC = timezone.utc
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "glm-genesis-monitor (research; github pages)"


def _get(url, timeout=60):
    last = None
    for _ in range(3):
        try:
            r = _SESSION.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as err:  # retry transient failures
            last = err
    raise RuntimeError(f"could not fetch {url}: {last}")


# ------------------------------------------------------------- parsing -----
def _latlon(token):
    """'165N' -> 16.5, '788W' -> -78.8 (ATCF stores tenths of a degree)."""
    token = token.strip()
    value = float(token[:-1]) / 10.0
    return -value if token[-1] in "SW" else value


@dataclass
class Track:
    storm_id: str                      # e.g. al922026, al092021, alA62024
    times: np.ndarray                  # datetime64[s], sorted
    lat: np.ndarray
    lon: np.ndarray
    vmax: np.ndarray
    mslp: np.ndarray
    kind: list                         # ATCF type per fix: DB, LO, WV, TD, TS, HU ...
    name: str = ""
    transitioned_from: str = ""        # 'alb92021' if the b-deck says "TRANSITIONED, alB92021 to al092021"
    transitioned_to: str = ""
    source: str = ""                   # set when positions do not come from this id's own b-deck alone
    precursor: str = ""                # id of the archived invest joined on, if any
    _t: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self._t = self.times.astype("datetime64[s]").astype(np.int64)

    # -- identity --
    @property
    def basin(self):
        return self.storm_id[:2].upper()

    @property
    def number(self):
        return self.storm_id[2:4]

    @property
    def year(self):
        return int(self.storm_id[4:8])

    @property
    def is_invest(self):
        num = self.number
        return (not num.isdigit()) or int(num) >= 90

    @property
    def first(self):
        return _to_dt(self.times[0])

    @property
    def last(self):
        return _to_dt(self.times[-1])

    # -- genesis --
    def genesis_time(self):
        """Time of the first record of a genesis type, or None."""
        for wanted in (config.GENESIS_TYPES, config.GENESIS_FALLBACK_TYPES):
            for t, k in zip(self.times, self.kind):
                if k in wanted:
                    return _to_dt(t)
        return None

    # -- centre positions --
    def centre(self, when, extrap_h=0.0):
        """Linear-in-time centre at each time in `when` (datetime64 array).

        Returns lat, lon, extrapolated. Positions before the first fix, or more
        than `extrap_h` hours past the last fix, are NaN. Past the last fix the
        centre moves with the motion over the previous ~12 h.
        """
        t = np.asarray(when).astype("datetime64[s]").astype(np.int64).astype(np.float64)
        lat = np.interp(t, self._t, self.lat, left=np.nan, right=np.nan)
        lon = np.interp(t, self._t, self.lon, left=np.nan, right=np.nan)
        extrap = np.zeros(t.shape, dtype=bool)
        after = t > self._t[-1]
        if extrap_h > 0 and after.any() and len(self._t) > 1:
            back = np.searchsorted(self._t, self._t[-1] - 12 * 3600, side="left")
            back = min(back, len(self._t) - 2)
            span = float(self._t[-1] - self._t[back])
            dlat = (self.lat[-1] - self.lat[back]) / span
            dlon = (self.lon[-1] - self.lon[back]) / span
            ahead = t - self._t[-1]
            ok = after & (ahead <= extrap_h * 3600)
            lat[ok] = self.lat[-1] + dlat * ahead[ok]
            lon[ok] = self.lon[-1] + dlon * ahead[ok]
            extrap[ok] = True
        elif extrap_h > 0 and after.any():
            ok = after & (t - self._t[-1] <= extrap_h * 3600)
            lat[ok], lon[ok] = self.lat[-1], self.lon[-1]
            extrap[ok] = True
        return lat, lon, extrap


def _to_dt(t64):
    return datetime.fromtimestamp(int(np.datetime64(t64, "s").astype(np.int64)), UTC)


def parse_bdeck(text, storm_id):
    """Turn b-deck text into a Track. Raises ValueError if no usable fixes."""
    rows = {}
    name = ""
    moved = moved_to = ""
    for line in text.splitlines():
        p = [s.strip() for s in line.split(",")]
        if len(p) < 11 or p[4] != "BEST":
            continue
        try:
            when = datetime.strptime(p[2], "%Y%m%d%H").replace(tzinfo=UTC)
            if p[3].isdigit():
                when += timedelta(minutes=int(p[3]))
            lat, lon = _latlon(p[6]), _latlon(p[7])
        except (ValueError, IndexError):
            continue
        if lat == 0.0 and lon == 0.0:
            continue
        vmax = float(p[8]) if p[8].lstrip("-").isdigit() else np.nan
        mslp = float(p[9]) if p[9].lstrip("-").isdigit() else np.nan
        if len(p) > 27 and p[27]:
            name = p[27]
        m = re.search(r"TRANSITIONED,\s*(\w+)\s+to\s+(\w+)", line)
        if m:
            moved, moved_to = m.group(1).lower(), m.group(2).lower()
        # several lines share a time (one per wind radius); keep the first
        rows.setdefault(when, (lat, lon, vmax if vmax > 0 else np.nan, mslp if mslp > 0 else np.nan, p[10]))
    if not rows:
        raise ValueError(f"no BEST records in b-deck for {storm_id}")
    order = sorted(rows)
    vals = [rows[k] for k in order]
    return Track(
        storm_id=storm_id,
        times=np.array([np.datetime64(int(k.timestamp()), "s") for k in order]),
        lat=np.array([v[0] for v in vals]),
        lon=np.array([v[1] for v in vals]),
        vmax=np.array([v[2] for v in vals]),
        mslp=np.array([v[3] for v in vals]),
        kind=[v[4] for v in vals],
        name=name.upper(),
        transitioned_from=moved,
        transitioned_to=moved_to,
    )


# ------------------------------------------------------------ fetching -----
def list_realtime():
    """Storm ids with a b-deck in the real-time directory, e.g. ['al922026', ...]."""
    html = _get(f"{config.ATCF_URL}/btk/")
    if html is None:
        raise RuntimeError("ATCF btk directory not found")
    text = html.decode("latin-1")
    ids = set()
    for basin, num, year in re.findall(r"\bb([a-z]{2})(\d{2})(\d{4})\.dat\b", text):
        if basin in config.BASINS:
            ids.add(f"{basin}{num}{year}")
    if not ids and "<a " not in text.lower():
        # not a directory listing at all: stop rather than treat every system as gone
        raise RuntimeError("ATCF btk directory listing could not be read")
    return sorted(ids)


def fetch_track(storm_id):
    """Fetch and parse a b-deck from the real-time directory or the archive.

    storm_id examples: al922026 (live invest), al092021 (named storm),
    alA62024 (archived invest; NHC replaces the 9 with a letter per reuse).
    """
    sid = storm_id[:2].lower() + storm_id[2:]
    year = sid[4:8]
    candidates = [
        f"{config.ATCF_URL}/btk/b{sid}.dat",
        f"{config.ATCF_URL}/archive/{year}/b{sid}.dat.gz",
        f"{config.ATCF_URL}/archive/{year}/invests/b{sid}.dat.gz",
    ]
    for url in candidates:
        raw = _get(url)
        if raw is None:
            continue
        if url.endswith(".gz"):
            raw = gzip.decompress(raw)
        return parse_bdeck(raw.decode("latin-1"), sid)
    raise FileNotFoundError(f"no b-deck found for {storm_id}")


# ------------------------------------------------ pre-genesis positions -----
# NHC's archived b-deck for a named system is the final best track, which
# starts at or shortly before genesis. The 72 h before that were tracked under
# an invest number, kept separately in archive/<year>/invests/. (Real-time
# b-decks still carry the invest history, so this only matters for past cases.)

def list_archive_invests(year, basin):
    """Ids of archived invests for a basin and year, e.g. ['alA02021', 'alA12021', ...]."""
    html = _get(f"{config.ATCF_URL}/archive/{year}/invests/")
    if html is None:
        return []
    found = re.findall(rf"\bb({basin.lower()}[A-Za-z]\d{year})\.dat\.gz\b", html.decode("latin-1"))
    return sorted(set(found))


def list_archive_numbered(year, basin):
    """Ids of numbered systems for a basin and year, from the archive or, for the current season, the real-time directory."""
    html = _get(f"{config.ATCF_URL}/archive/{year}/")
    nums = re.findall(rf"\bb{basin.lower()}(\d\d){year}\.dat\.gz\b", html.decode("latin-1")) if html else []
    if not nums:
        nums = [sid[2:4] for sid in list_realtime() if sid.startswith(basin.lower()) and sid.endswith(str(year))]
    return sorted({f"{basin.lower()}{n}{year}" for n in nums if int(n) < 50})


def load_archive_invests(year, basin):
    """{id: Track} for every archived invest of a basin and year that can be read."""
    out = {}
    for iid in list_archive_invests(year, basin):
        try:
            raw = _get(f"{config.ATCF_URL}/archive/{year}/invests/b{iid}.dat.gz")
            out[iid] = parse_bdeck(gzip.decompress(raw).decode("latin-1"), iid)
        except Exception:
            continue
    return out


def find_precursors(track, invests):
    """Archived invests that are the earlier life of a numbered system, best match first.

    An invest qualifies if it has a fix within 12 h of where the system's own
    track begins and that fix is within 300 km of it, or, whatever the
    distance, the b-decks carry a TRANSITIONED note linking the two. The time
    test applies to notes as well: the letter NHC gives an archived invest
    does not always agree with the one in the note, and an invest from weeks
    earlier must never be joined on. Returns [(km, Track)], km = -1 for a note.
    """
    if track.is_invest:
        return []
    start_s = track._t[0]
    found = []
    for iid, inv in invests.items():
        k = int(np.argmin(np.abs(inv._t - start_s)))    # invest fix nearest the start of the system's track
        if abs(float(inv._t[k] - start_s)) > 12 * 3600:
            continue
        if inv.transitioned_to == track.storm_id.lower() or track.transitioned_from == iid.lower():
            found.append((-1.0, inv))
            continue
        if inv._t[-1] > start_s + 48 * 3600:
            continue                                   # it carried on for days as something else
        when = np.array([inv._t[k]], dtype="int64").astype("datetime64[s]")
        lat, lon, _ = track.centre(when)
        if not np.isfinite(lat[0]):                     # the invest ends before the track starts
            lat, lon = track.lat[:1], track.lon[:1]
        dist = float(_haversine(inv.lat[k], inv.lon[k], lat[0], lon[0]))
        if dist <= 300.0:
            found.append((dist, inv))
    return sorted(found, key=lambda f: f[0])


def _merge(early, late, source):
    """`late` with the fixes of `early` that precede its first fix put in front."""
    keep = early._t < late._t[0]
    return Track(
        storm_id=late.storm_id,
        times=np.concatenate([early.times[keep], late.times]),
        lat=np.concatenate([early.lat[keep], late.lat]),
        lon=np.concatenate([early.lon[keep], late.lon]),
        vmax=np.concatenate([early.vmax[keep], late.vmax]),
        mslp=np.concatenate([early.mslp[keep], late.mslp]),
        kind=[k for k, w in zip(early.kind, keep) if w] + list(late.kind),
        name=late.name, transitioned_from=late.transitioned_from, transitioned_to=late.transitioned_to,
        source=source, precursor=early.storm_id,
    )


def with_precursor(track, needed_from, invests=None, log=print):
    """Extend an archived numbered system back in time with the invest it grew out of.

    `invests` is the result of load_archive_invests(); it is fetched if not given.
    """
    if track.is_invest or track.first <= needed_from:
        return track
    if invests is None:
        invests = load_archive_invests(track.year, track.storm_id[:2])
    usable = [(score, inv) for score, inv in find_precursors(track, invests) if inv._t[0] < track._t[0]]
    if not usable:
        log(f"  {track.storm_id}: no archived invest matches the start of its track; "
            f"positions begin {track.first:%Y-%m-%d %HZ}")
        return track
    score, inv = usable[0]
    how = "TRANSITIONED note" if score < 0 else f"{score:.0f} km from the first position"
    merged = _merge(inv, track, f"Positions before {track.first:%d %b %HZ} are from invest {inv.storm_id} in the NHC invest archive.")
    log(f"  {track.storm_id}: joined invest {inv.storm_id} ({how}); positions now begin {merged.first:%Y-%m-%d %HZ}")
    return merged


def _haversine(lat1, lon1, lat2, lon2):
    from .geo import haversine
    return haversine(lat1, lon1, lat2, lon2)


def load_best_nc(path, storm_id):
    """Read a combined invest + best-track NetCDF file (the BEST_b<id>.nc files in the paper's Misc_files).

    Uses initialization_time, track_lat, track_lon (0-360 or -180-180), track_type,
    and track_maxwind / track_mslp when present.
    """
    import xarray as xr
    with xr.open_dataset(path) as ds:
        t = ds["initialization_time"].values.astype("datetime64[s]")
        lat = ds["track_lat"].values.astype(float)
        lon = ds["track_lon"].values.astype(float)
        kind = [k.decode() if isinstance(k, bytes) else str(k) for k in ds["track_type"].values]
        vmax = ds["track_maxwind"].values.astype(float) if "track_maxwind" in ds else np.full(len(t), np.nan)
        mslp = ds["track_mslp"].values.astype(float) if "track_mslp" in ds else np.full(len(t), np.nan)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    ok = np.isfinite(lat) & np.isfinite(lon)
    order = np.argsort(t[ok])
    pick = lambda a: np.asarray(a)[ok][order]
    return Track(storm_id=storm_id[:2].lower() + storm_id[2:], times=pick(t), lat=pick(lat), lon=pick(lon),
                 vmax=pick(vmax), mslp=pick(mslp), kind=[k.strip() for k in pick(np.array(kind, dtype=object))],
                 source="Positions are from the combined invest and best-track file used in the paper.")

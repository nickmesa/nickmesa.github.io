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
    transitioned_from: str = ""
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
    moved = ""
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
            moved = m.group(1).lower()
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

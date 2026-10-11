"""A local stand-in for the NHC ATCF site and the NOAA GOES S3 buckets.

Serves b-deck text and GLM L2 LCFA netCDF files written with the same
variable names, packing and key layout as the real ones, so the pipeline runs
through its real download and read code in tests. Lightning in these files is
made up; nothing here is an observation.
"""
import gzip
import re
import threading
from datetime import timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import netCDF4
import numpy as np

UTC = timezone.utc
ENERGY_SCALE, ENERGY_OFFSET = 9.99996e-16, 2.8515e-16      # J, as in operational files
AREA_SCALE = 152601.9                                      # m2


def quantize_energy(joules):
    """Energy as it comes back out of the packed 16-bit variable."""
    return np.round((np.asarray(joules, float) - ENERGY_OFFSET) / ENERGY_SCALE) * ENERGY_SCALE + ENERGY_OFFSET


def quantize_area(m2):
    return np.round(np.asarray(m2, float) / AREA_SCALE) * AREA_SCALE


def lcfa_bytes(lat, lon, energy_j, area_m2, flag=None, area_units="m2"):
    """One LCFA file holding the given groups."""
    n = len(lat)
    flag = np.zeros(n, int) if flag is None else np.asarray(flag)
    ds = netCDF4.Dataset("fake.nc", mode="w", memory=4096)
    ds.createDimension("number_of_groups", n)
    ds.createDimension("number_of_flashes", 0)
    v = ds.createVariable("group_lat", "f4", ("number_of_groups",))
    v.units = "degrees_north"
    v[:] = np.asarray(lat, "f4")
    v = ds.createVariable("group_lon", "f4", ("number_of_groups",))
    v.units = "degrees_east"
    v[:] = np.asarray(lon, "f4")
    v = ds.createVariable("group_energy", "i2", ("number_of_groups",), fill_value=-1)
    v.set_auto_maskandscale(False)
    v.setncattr("_Unsigned", "true")
    v.scale_factor, v.add_offset, v.units = ENERGY_SCALE, ENERGY_OFFSET, "J"
    v[:] = np.round((np.asarray(energy_j, float) - ENERGY_OFFSET) / ENERGY_SCALE).astype(np.uint16).view(np.int16)
    v = ds.createVariable("group_area", "i2", ("number_of_groups",), fill_value=-1)
    v.set_auto_maskandscale(False)
    v.setncattr("_Unsigned", "true")
    scale = AREA_SCALE if area_units == "m2" else AREA_SCALE / 1e6
    v.scale_factor, v.add_offset, v.units = scale, 0.0, area_units
    v[:] = np.round(np.asarray(area_m2, float) / AREA_SCALE).astype(np.uint16).view(np.int16)
    v = ds.createVariable("group_quality_flag", "i2", ("number_of_groups",), fill_value=-1)
    v.set_auto_maskandscale(False)
    v.setncattr("_Unsigned", "true")
    v.flag_values = np.array([0, 1, 3, 5], "i2")
    v[:] = flag.astype(np.int16)
    return bytes(ds.close())


def lcfa_key(sat, start):
    end = start + timedelta(seconds=20)
    stamp = lambda t: t.strftime("%Y%j%H%M%S") + "0"
    return (f"GLM-L2-LCFA/{start:%Y}/{start:%j}/{start:%H}/"
            f"OR_GLM-L2-LCFA_G{sat}_s{stamp(start)}_e{stamp(end)}_c{stamp(end + timedelta(seconds=2))}.nc")


def bdeck_text(basin, number, fixes, name="INVEST"):
    """fixes: [(datetime, lat, lon, vmax, mslp, type)] -> b-deck text in ATCF layout."""
    lines = []
    for when, lat, lon, vmax, mslp, kind in fixes:
        la = f"{abs(round(lat * 10)):d}{'N' if lat >= 0 else 'S'}"
        lo = f"{abs(round(lon * 10)):d}{'W' if lon < 0 else 'E'}"
        radii = [34, 50] if vmax >= 50 else [34] if vmax >= 34 else [0]      # one line per wind radius
        for rad in radii:
            lines.append(
                f"{basin.upper()}, {number}, {when:%Y%m%d%H},   , BEST,   0, {la:>4}, {lo:>5}, {vmax:3d}, {mslp:4d}, {kind},"
                f" {rad:3d}, NEQ,    0,    0,    0,    0, 1012,  150,  60,   0,   0,   L,   0,    ,   0,   0,"
                f" {name:>10}, S,  0,    ,    0,    0,    0,    0, genesis-num, 040, ")
    return "\n".join(lines) + "\n"


class FakeNOAA:
    """In-memory files plus an HTTP server on localhost."""

    def __init__(self):
        self.btk = {}            # 'bal922026.dat' -> text
        self.archive = {}        # '2024/bal022024.dat.gz' or '2024/invests/balA62024.dat.gz' -> text
        self.objects = {}        # bucket -> {key: bytes}
        self.gets = []           # keys of GLM files downloaded
        self.lists = 0
        self.page_size = 100     # small, so listing pagination is exercised
        self._server = None

    # -- content --
    def put_glm(self, bucket, start, **groups):
        sat = bucket[-2:]
        self.objects.setdefault(bucket, {})[lcfa_key(sat, start)] = lcfa_bytes(**groups)

    # -- server --
    def start(self):
        store = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype="application/octet-stream", code=200):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                parts = url.path.strip("/").split("/")
                if parts[0] == "atcf":
                    rest = "/".join(parts[1:])
                    if rest == "btk":
                        links = "".join(f'<a href="{n}">{n}</a>\n' for n in sorted(store.btk))
                        return self._send(f"<html><body>{links}</body></html>".encode(), "text/html")
                    if rest.startswith("btk/") and rest[4:] in store.btk:
                        return self._send(store.btk[rest[4:]].encode(), "text/plain")
                    m = re.fullmatch(r"archive/(\d{4})/invests", rest)
                    if m:
                        names = sorted(k.split("/")[-1] for k in store.archive if k.startswith(f"{m.group(1)}/invests/"))
                        links = "".join(f'<a href="{n}">{n}</a>\n' for n in names)
                        return self._send(f"<html><body>{links}</body></html>".encode(), "text/html")
                    if rest.startswith("archive/") and rest[8:] in store.archive:
                        return self._send(gzip.compress(store.archive[rest[8:]].encode()))
                    return self._send(b"not found", "text/plain", 404)
                bucket = parts[0]
                objects = store.objects.get(bucket, {})
                if len(parts) == 1:        # S3 ListObjectsV2
                    q = parse_qs(url.query)
                    prefix = q.get("prefix", [""])[0]
                    keys = sorted(k for k in objects if k.startswith(prefix))
                    begin = int(q.get("continuation-token", ["0"])[0])
                    page = keys[begin:begin + store.page_size]
                    more = begin + store.page_size < len(keys)
                    xml = "<ListBucketResult>" + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in page)
                    xml += f"<IsTruncated>{str(more).lower()}</IsTruncated>"
                    if more:
                        xml += f"<NextContinuationToken>{begin + store.page_size}</NextContinuationToken>"
                    store.lists += 1
                    return self._send((xml + "</ListBucketResult>").encode(), "application/xml")
                key = "/".join(parts[1:])
                if key in objects:
                    store.gets.append(key)
                    return self._send(objects[key])
                return self._send(b"not found", "text/plain", 404)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self):
        if self._server:
            self._server.shutdown()

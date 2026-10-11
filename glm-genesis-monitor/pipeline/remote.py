"""Read parts of a remote file with HTTP range requests.

ABI and ERA5 files are netCDF-4 (HDF5). A storm needs a few hundred kilobytes
of a 25 MB ABI file and one 65 MB chunk at a time of a 1.5 GB ERA5 file, so
the files are opened in place with h5py instead of being downloaded whole.

Always open with `with remote.open_h5(url) as (h5, raw): ...`. Leaving an HDF5
file open on a Python file object crashes the interpreter at exit.
"""
import io
import time
from contextlib import contextmanager

import requests

BLOCK = 1 << 17            # 128 KB: metadata reads are small and scattered
BIG = 1 << 22              # reads this large are passed straight through, not cached


class RangeFile(io.RawIOBase):
    """A read-only, seekable file object backed by HTTP range requests."""

    def __init__(self, url, session=None, block=BLOCK, timeout=120):
        super().__init__()
        self.url, self.block, self.timeout = url, block, timeout
        self.session = session or requests.Session()
        self.pos, self.blocks = 0, {}
        self.requests, self.bytes = 0, 0
        first, total = self._get(0, block - 1, want_total=True)
        self.size = total
        self.blocks[0] = first

    # -- http --
    def _get(self, lo, hi, want_total=False):
        last = None
        for attempt in range(4):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=self.timeout)
                if r.status_code in (403, 404):
                    raise FileNotFoundError(f"{self.url}: HTTP {r.status_code}")
                if r.status_code == 416:                     # range starts past the end of the file
                    return (b"", lo) if want_total else b""
                r.raise_for_status()
                body = r.content
                self.requests += 1
                self.bytes += len(body)
                if not want_total:
                    return body if r.status_code == 206 else body[lo:hi + 1]
                if r.status_code == 206:
                    return body, int(r.headers["Content-Range"].rsplit("/", 1)[1])
                return body[:hi + 1], len(body)              # the server ignored the range and sent everything
            except FileNotFoundError:
                raise
            except (requests.RequestException, KeyError, ValueError) as err:
                last = err
                time.sleep(1.5 * (attempt + 1))
        raise OSError(f"could not read {self.url}: {last}")

    # -- file interface --
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _span(self, lo, hi):
        """Bytes lo..hi-1, fetching whichever blocks are not held yet in one request."""
        if hi - lo >= BIG:
            return self._get(lo, hi - 1)
        b0, b1 = lo // self.block, (hi - 1) // self.block
        missing = [b for b in range(b0, b1 + 1) if b not in self.blocks]
        if missing:
            first, last = missing[0], missing[-1]
            data = self._get(first * self.block, min((last + 1) * self.block, self.size) - 1)
            for b in range(first, last + 1):
                self.blocks[b] = data[(b - first) * self.block:(b - first + 1) * self.block]
        joined = b"".join(self.blocks[b] for b in range(b0, b1 + 1))
        return joined[lo - b0 * self.block:hi - b0 * self.block]

    def readinto(self, buf):
        view = memoryview(buf)
        n = max(0, min(len(view), self.size - self.pos))
        if n:
            view[:n] = self._span(self.pos, self.pos + n)
            self.pos += n
        return n


@contextmanager
def open_h5(url, session=None):
    """Open a remote HDF5/netCDF-4 file. Yields (h5py.File, RangeFile) and always closes both."""
    import h5py                       # imported here so only the processes that read these files load HDF5

    raw = RangeFile(url, session=session)
    h5 = h5py.File(raw, "r")
    try:
        yield h5, raw
    finally:
        h5.close()
        raw.close()

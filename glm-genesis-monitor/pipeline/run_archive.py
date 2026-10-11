"""Process a past disturbance and add it to the archive.

    python -m pipeline.run_archive --id al092021
    python -m pipeline.run_archive --id alA62024 --start 2024-06-28T12 --end 2024-07-03T12

--id is an ATCF id:
    al092021   a numbered system (b-deck from the NHC archive). The window is
               the 72 h before genesis, as in the paper.
    alA62024   an archived invest. NHC replaces the 9 of the invest number with
               a letter for each reuse in a season (A6 = first 96L, B6 = second).
               See https://ftp.nhc.noaa.gov/atcf/archive/<year>/invests/.
               The window is the invest's whole track unless --start/--end are
               given.

A long window is thousands of GLM files. The run can be interrupted and
restarted; finished 5-minute steps are kept in the cache folder.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

from . import atcf, config, glm, process

UTC = timezone.utc


def log(msg):
    print(msg, flush=True)


def run(storm_id, start=None, end=None, now=None):
    now = now or datetime.now(UTC)
    track = atcf.fetch_track(storm_id)
    w_start, w_end, genesis = process.window_for(track)
    start, end = start or w_start, end or w_end
    if genesis is not None and not (start <= genesis <= end):
        genesis = None
    steps = process.Steps(track, start, end)
    if not len(steps):
        raise SystemExit(f"{storm_id}: the b-deck has no positions between {start} and {end}")
    key = track.storm_id
    cache = glm.CellCache(process.cache_path(key))
    need = process.missing(cache, steps, now)
    log(f"{key} {process.storm_label(track)}: {start:%Y-%m-%d %HZ} to {end:%Y-%m-%d %HZ}, "
        f"{len(steps)} steps, {len(need)} to download")
    # save progress every 12 h of data so an interrupted run can resume
    block = 144
    for lo in range(0, len(need), block):
        glm.fill({key: (cache, need[lo:lo + block])}, log=log)
        cache.save()
    cache.keep_only(steps.t)
    feats, info = process.analyse(steps, cache, log=log)
    outcome = "developed" if genesis is not None else "nondeveloper"
    product = process.build_product(key, track, steps, cache, feats, info, status="ended", outcome=outcome,
                                    genesis=genesis, now=now)
    index = process.load_index()
    entry = process.publish(product, index)
    process.save_index(index, now)
    if os.path.exists(cache.path):
        os.remove(cache.path)
    log(f"{key}: {entry['n_features']} features and {entry['n_groups']} groups within {config.RADIUS_KM:.0f} km (away from land)")
    return product


def _when(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC) if text else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="ATCF id(s), comma separated")
    ap.add_argument("--start", help="window start, UTC ISO time (default: 72 h before genesis, or first fix)")
    ap.add_argument("--end", help="window end, UTC ISO time (default: genesis, or last fix)")
    args = ap.parse_args(argv)
    for sid in [s.strip() for s in args.id.split(",") if s.strip()]:
        run(sid, _when(args.start), _when(args.end))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Add every disturbance of a past season to the archive.

    python -m pipeline.run_season --year 2022 --basin al --dry-run     # write the plan only
    python -m pipeline.run_season --year 2022 --basin al               # process it

What a season holds:
  * every numbered system (depressions, storms, hurricanes, potential tropical
    cyclones). One that reached genesis gets the LOOKBACK_H hours before it;
    one that never did gets its whole track. Either way the invest it grew
    out of is joined on, because the archived best track starts near genesis.
  * every archived invest that never became a numbered system, as a
    non-developer over its whole track.

The plan is written to <data>/backfill/<year>-<basin>.json and updated as cases
finish, so progress and the reason for every skipped or failed case are on
record. Cases already in the archive are left alone unless --redo is given,
which makes an interrupted season safe to run again.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from . import atcf, config, process, run_archive
from .geo import haversine

UTC = timezone.utc
MIN_INVEST_H = 12          # shorter invest tracks are not worth a record
MIN_WINDOW_H = 1           # a developed system needs at least this much track before genesis
ACTIVE_DAYS = 5            # an unfinished system this recent is left to the hourly run


def log(msg):
    print(msg, flush=True)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%MZ") if dt is not None else None


def _case(track, kind):
    return {"id": track.storm_id, "kind": kind, "basin": track.storm_id[:2], "label": process.storm_label(track),
            "outcome": None, "genesis": None, "start": None, "end": None, "hours": None, "precursor": "",
            "status": "pending", "reason": ""}


def _skip(case, reason):
    case["status"], case["reason"] = "skipped", reason
    return case


def build_plan(year, now=None, basins=config.BASINS):
    """(cases, invests) for one season, all basins together.

    The basins are planned together because a disturbance can be an invest in
    one basin and a numbered system in the other.
    """
    now = now or datetime.now(UTC)
    invests, numbered = {}, []
    for basin in basins:
        invests.update(atcf.load_archive_invests(year, basin))
        numbered += atcf.list_archive_numbered(year, basin)

    cases, used, origin = [], {}, []        # origin: (id, genesis, lat, lon) of the systems kept
    for sid in numbered:
        try:
            track = atcf.fetch_track(sid)
        except Exception as err:
            cases.append({"id": sid, "kind": "numbered", "basin": sid[:2], "label": sid.upper(), "outcome": None,
                          "genesis": None, "start": None, "end": None, "hours": None, "precursor": "",
                          "status": "failed", "reason": f"could not read the b-deck: {err}"[:200]})
            continue
        for _, inv in atcf.find_precursors(track, invests):
            used.setdefault(inv.storm_id, track.storm_id)
        genesis = track.genesis_time()
        same = _same_system(track, genesis, origin)
        if same:
            # a storm that crosses from one basin to the other is given a second number there,
            # and that b-deck repeats the whole track
            case = _case(track, "numbered")
            case.update(outcome="developed", genesis=_iso(genesis))
            cases.append(_skip(case, f"same system as {same} (crossed basins)"))
            continue
        if genesis is not None:
            g = np.array([np.datetime64(genesis.replace(tzinfo=None), "s")])
            lat, lon, _ = track.centre(g)
            origin.append((track.storm_id, genesis, float(lat[0]), float(lon[0])))
        anchor = genesis or track.first
        merged = atcf.with_precursor(track, anchor - timedelta(hours=config.LOOKBACK_H), invests=invests, log=lambda m: None)
        start, end, genesis = process.window_for(merged)
        case = _case(merged, "numbered")
        case.update(outcome="developed" if genesis is not None else "nondeveloper", genesis=_iso(genesis),
                    start=_iso(start), end=_iso(end), hours=round((end - start).total_seconds() / 3600, 1),
                    precursor=merged.precursor)
        if genesis is None and now - track.last < timedelta(days=ACTIVE_DAYS):
            _skip(case, "may still be active; the hourly run follows it")
        elif genesis is not None and case["hours"] < MIN_WINDOW_H:
            _skip(case, "the best track starts at genesis and no archived invest matches it")
        cases.append(case)

    seen = {}
    for iid, inv in sorted(invests.items()):
        start, end, _ = process.window_for(inv)
        case = _case(inv, "invest")
        case.update(outcome="nondeveloper", start=_iso(start), end=_iso(end),
                    hours=round((end - start).total_seconds() / 3600, 1))
        became = used.get(iid) or (inv.transitioned_to if inv.transitioned_to and not _is_invest_id(inv.transitioned_to) else "")
        shape = (int(inv._t[0]), int(inv._t[-1]), len(inv._t), round(float(inv.lat[0]), 1), round(float(inv.lon[0]), 1))
        if became:
            _skip(case, f"became {became}")
        elif case["hours"] < MIN_INVEST_H:
            _skip(case, f"track shorter than {MIN_INVEST_H} h")
        elif shape in seen:
            _skip(case, f"same track as {seen[shape]}")       # NHC's archive holds some invests under two letters
        else:
            seen[shape] = iid
        cases.append(case)
    return cases, invests


def _same_system(track, genesis, origin):
    """Id of an earlier numbered system that is this one under another basin's number, or ''."""
    if genesis is None:
        return ""
    g = np.array([np.datetime64(genesis.replace(tzinfo=None), "s")])
    lat, lon, _ = track.centre(g)
    for sid, when, la, lo in origin:
        if sid[:2] != track.storm_id[:2] and when == genesis and haversine(la, lo, float(lat[0]), float(lon[0])) < 100.0:
            return sid
    return ""


def _is_invest_id(sid):
    num = sid[2:4]
    return (not num.isdigit()) or int(num) >= 90


def plan_path(year, basin):
    return os.path.join(config.DATA_DIR, "backfill", f"{year}-{basin}.json")


def save_plan(year, basin, cases, now):
    counts = {}
    for c in cases:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    process._write_json(plan_path(year, basin), {"year": year, "basin": basin, "updated": _iso(now),
                                                 "counts": counts, "cases": cases})


def run(year, basin, dry_run=False, redo=False, max_hours=5.0, now=None):
    now = now or datetime.now(UTC)
    began = time.monotonic()
    all_cases, invests = build_plan(year, now)
    cases = [c for c in all_cases if c["basin"] == basin]
    have = {e["id"] for e in process.load_index()["storms"] if e["status"] == "ended" and not e.get("pending_steps")}
    for c in cases:
        if c["status"] == "pending" and c["id"] in have and not redo:
            c["status"], c["reason"] = "done", "already in the archive"
    save_plan(year, basin, cases, now)
    todo = [c for c in cases if c["status"] == "pending"]
    log(f"{year} {basin}: {len(cases)} b-decks, {len(todo)} to process, "
        f"{sum(c['hours'] for c in todo):.0f} h of GLM data")
    for c in cases:
        log(f"  {c['id']}  {c['label']:<24} {c['outcome'] or '-':<13} {c['start']} to {c['end']}  "
            f"{c['status']}{' (' + c['reason'] + ')' if c['reason'] else ''}"
            f"{' + ' + c['precursor'] if c['precursor'] else ''}")
    if dry_run:
        return cases

    for c in todo:
        if time.monotonic() - began > max_hours * 3600:
            log(f"stopping after {max_hours} h; run the season again to finish the rest")
            break
        try:
            product = run_archive.run(c["id"], now=now, invests=invests)
            feats = product["features"]
            c.update(status="done", reason="",
                     n_features=int(sum(1 for landed in feats["land"] if not landed)), n_steps=product["steps"]["n"])
        except Exception as err:
            c.update(status="failed", reason=f"{type(err).__name__}: {err}"[:300])
            log(f"{c['id']}: FAILED {c['reason']}")
        save_plan(year, basin, cases, datetime.now(UTC))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--basin", required=True, choices=config.BASINS)
    ap.add_argument("--dry-run", action="store_true", help="write the plan and stop")
    ap.add_argument("--redo", action="store_true", help="process cases that are already in the archive again")
    ap.add_argument("--max-hours", type=float, default=5.0, help="stop starting new cases after this long")
    args = ap.parse_args(argv)
    cases = run(args.year, args.basin, dry_run=args.dry_run, redo=args.redo, max_hours=args.max_hours)
    failed = [c["id"] for c in cases if c["status"] == "failed"]
    if failed:
        log(f"failed: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

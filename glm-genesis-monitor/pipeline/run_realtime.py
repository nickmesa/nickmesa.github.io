"""Hourly real-time update.

    python -m pipeline.run_realtime [--now 2026-10-11T06:00]

Finds every Atlantic and East Pacific disturbance in the NHC real-time b-deck
directory, downloads the GLM files it does not have yet, reruns the tobac
analysis and rewrites the JSON the page reads.

What is monitored:
  * invests (numbers 90-99) with a fix in the last STALE_H hours
  * numbered systems that have not reached genesis yet (depressions and
    potential tropical cyclones)
A numbered system is closed out at genesis with the LOOKBACK_H hours before
it. A disturbance that stops being tracked without reaching genesis is closed
out as a non-developer.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from . import atcf, config, glm, process
from .geo import haversine

UTC = timezone.utc
MAX_STEPS_PER_RUN = 1500      # ~22,500 files; a larger backlog is spread over later runs


def log(msg):
    print(msg, flush=True)


def key_for(track, index):
    """Stable page id for a live b-deck.

    Numbered systems keep their ATCF id. Invest numbers are reused within a
    season, so an invest is keyed by its id plus the date of its first fix
    when it was first seen, e.g. al922026_1006.
    """
    if not track.is_invest:
        return track.storm_id
    for e in index["storms"]:
        if e["atcf_id"] == track.storm_id and e["status"] == "active" and e["t_end"] is not None:
            # same number, and the track picks up where the stored record left off
            if track.first.timestamp() <= e["t_end"] + 24 * 3600:
                return e["id"]
    return f"{track.storm_id}_{track.first:%m%d}"


def follows(track, t, lat, lon):
    """True if `track` passes within 100 km (median) of the given fixes."""
    t = np.asarray(t, dtype="int64").astype("datetime64[s]")
    clat, clon, _ = track.centre(t)
    ok = np.isfinite(clat)
    if ok.sum() < 2:
        return False
    d = haversine(clat[ok], clon[ok], np.asarray(lat, float)[ok], np.asarray(lon, float)[ok])
    return float(np.median(d)) < 100.0


def stored_product(key):
    path = os.path.join(config.DATA_DIR, "storms", key + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--now", help="pretend it is this UTC time (ISO), for testing and reruns")
    args = ap.parse_args(argv)
    now = datetime.fromisoformat(args.now).replace(tzinfo=UTC) if args.now else datetime.now(UTC)
    data_end = now - timedelta(minutes=config.LATENCY_MIN)

    index = process.load_index()
    log(f"real-time update for {now:%Y-%m-%d %H:%MZ}")

    # ---- 1. which b-decks matter ------------------------------------------
    jobs, unreadable = [], set()
    for sid in atcf.list_realtime():
        try:
            track = atcf.fetch_track(sid)
        except Exception as err:
            log(f"  {sid}: skipped ({err})")
            unreadable.add(sid)
            continue
        age_h = (now - track.last).total_seconds() / 3600
        genesis = None if track.is_invest else track.genesis_time()
        key = key_for(track, index)
        known = next((e for e in index["storms"] if e["id"] == key), None)

        if genesis is not None:
            finished = known is not None and known["outcome"] == "developed" and known["pending_steps"] == 0
            recent = (now - genesis) <= timedelta(days=config.AUTO_ARCHIVE_DAYS)
            if finished or not (recent or known):
                continue
            start, end, _ = process.window_for(track)
            steps = process.Steps(track, start, end)
            job = dict(status="ended", outcome="developed")
        elif age_h <= config.STALE_H:
            start = max(track.first, data_end - timedelta(hours=config.MAX_LIFETIME_H))
            steps = process.Steps(track, start, data_end, extrap_h=config.MAX_EXTRAP_H)
            job = dict(status="active", outcome="invest" if track.is_invest else "pre-genesis")
        else:
            continue          # stale and never reached genesis: closed out in step 4
        if len(steps):
            jobs.append(dict(job, key=key, track=track, steps=steps, genesis=genesis, note=known["note"] if known else "",
                             cache=glm.CellCache(process.cache_path(key))))

    # ---- 2. invests that have been numbered are replaced by the numbered system
    superseded = {}       # invest key -> cache
    for j in [j for j in jobs if not j["track"].is_invest]:
        for o in [o for o in jobs if o["track"].is_invest and o["track"].basin == j["track"].basin]:
            if follows(j["track"], o["track"]._t, o["track"].lat, o["track"].lon):
                superseded[o["key"]] = o["cache"]
                j["note"] = f"Tracked as {process.storm_label(o['track'])} before it was numbered."
        for e in index["storms"]:
            is_invest = e["atcf_id"][2:4].isdigit() and int(e["atcf_id"][2:4]) >= 90
            if e["status"] != "active" or not is_invest or e["basin"] != j["track"].basin or e["id"] in superseded:
                continue
            old = stored_product(e["id"])
            if old and old["track"]["t"] and follows(j["track"], old["track"]["t"], old["track"]["lat"], old["track"]["lon"]):
                superseded[e["id"]] = glm.CellCache(process.cache_path(e["id"]))
                j["note"] = f"Tracked as {e['label']} before it was numbered."
    jobs = [j for j in jobs if j["key"] not in superseded]
    live = {j["key"] for j in jobs}

    # ---- 3. download what is missing ---------------------------------------
    wanted, budget = {}, MAX_STEPS_PER_RUN
    for j in sorted(jobs, key=lambda j: j["status"] != "active"):       # live systems first
        others = [o["cache"] for o in jobs if o is not j] + list(superseded.values())
        need = process.missing(j["cache"], j["steps"], data_end, others)
        take = need[:max(budget, 0)]
        budget -= len(take)
        if take:
            wanted[j["key"]] = (j["cache"], take)
        log(f"  {j['key']} {process.storm_label(j['track'])}: {len(j['steps'])} steps, {len(need)} to download"
            + (f" ({len(need) - len(take)} left for the next run)" if len(take) < len(need) else ""))
    if wanted:
        log(f"  downloaded {glm.fill(wanted, log=log, now=now)} GLM files")

    # ---- 4. analyse and publish --------------------------------------------
    for j in jobs:
        cache, steps = j["cache"], j["steps"]
        cache.keep_only(steps.t)
        feats, info = process.analyse(steps, cache, log=log)
        product = process.build_product(j["key"], j["track"], steps, cache, feats, info, status=j["status"],
                                        outcome=j["outcome"], genesis=j["genesis"], now=now, note=j["note"])
        entry = process.publish(product, index)
        log(f"  {j['key']}: {entry['n_features']} features and {entry['n_groups']} groups within "
            f"{config.RADIUS_KM:.0f} km (away from land)"
            + (f"; {entry['pending_steps']} steps not downloaded yet" if entry["pending_steps"] else ""))
        if j["status"] == "ended" and entry["pending_steps"] == 0:
            if os.path.exists(cache.path):
                os.remove(cache.path)           # finished; nothing more to download
        else:
            cache.save()

    for key in superseded:
        process.remove(key, index)
        log(f"  {key}: replaced by its numbered system")

    # ---- 5. close out systems that are no longer tracked --------------------
    for e in index["storms"]:
        if e["status"] != "active" or e["id"] in live or e["atcf_id"] in unreadable:
            continue
        e["status"], e["outcome"] = "ended", "nondeveloper"
        prod = stored_product(e["id"])
        if prod:
            prod["status"], prod["outcome"] = "ended", "nondeveloper"
            process._write_json(os.path.join(config.DATA_DIR, "storms", e["id"] + ".json"), prod)
        if os.path.exists(process.cache_path(e["id"])):
            os.remove(process.cache_path(e["id"]))
        log(f"  {e['id']}: no longer tracked, archived as a non-developer")

    process.save_index(index, now)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

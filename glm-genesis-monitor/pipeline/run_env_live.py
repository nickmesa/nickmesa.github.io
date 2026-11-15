"""Keep the infrared layer of the real-time page current. Runs after the hourly lightning update.

    python -m pipeline.run_env_live [--now 2026-10-11T06:00]

For every record in the lightning index this writes the channel 13 images and
cold cloud fractions (and ERA5 vorticity once NCAR has published it) to
config.ENV_DIR, which the real-time workflow points at <data>/env/ on the
glm-data branch. The page reads it from there beside the lightning.

  * Active records are updated every run. Scans read earlier are kept in the
    cache (config.CACHE_DIR/env), so an update reads only the scans that have
    arrived since the last one.
  * A record that has ended and has no infrared in the archive branch
    (glm-env) is processed once, a few per run.
  * ERA5 is published months behind real time. Hours older than
    config.ERA5_MIN_AGE_DAYS are asked about once a day, and the vorticity is
    added to the record when they turn up.
  * A record whose infrared has since been archived on glm-env with at least as
    much vorticity, or that left the lightning index, is removed from here.

The archive branch is read only through its small index.json
(GLM_ARCHIVE_ENV_INDEX_URL). If that cannot be read, no backlog of ended
records is started.
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from . import config, env, process, run_env

UTC = timezone.utc


def log(msg):
    print(msg, flush=True)


def prior_path(key):
    return os.path.join(config.CACHE_DIR, "env", key + ".pkl.gz")


def load_prior(key):
    try:
        with gzip.open(prior_path(key), "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return {}                         # nothing stored, or unreadable: read the record from scratch


def save_prior(key, prior):
    path = prior_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path + ".tmp", "wb", compresslevel=3) as fh:
        pickle.dump(prior, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)


def archive_index(url=None):
    """The cases listed in the archive branch's env index, or None if that cannot be read."""
    url = url or os.environ.get("GLM_ARCHIVE_ENV_INDEX_URL")
    if not url:
        return None
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        cases = r.json()["cases"]
        return cases if isinstance(cases, dict) else None
    except Exception as err:
        log(f"  could not read the archive index ({type(err).__name__}); not starting any backlog")
        return None


def covered(a):
    return bool(a) and not a.get("error") and bool(a.get("frames"))


def plan(storms, live, archive, now):
    """Which records to update, in order, and which to remove. Pure, so it can be tested.

    storms: entries of the lightning index; live: cases of the live env index;
    archive: cases of the archive branch's env index, or None if unknown.
    """
    ids = {e["id"] for e in storms}
    remove = [k for k in live if k not in ids]
    active, ended = [], []
    for e in sorted(storms, key=lambda e: -(e.get("t_end") or 0)):
        key, have = e["id"], live.get(e["id"])
        if e["status"] == "active":
            active.append(key)
            continue
        there = archive.get(key) if archive is not None else None
        if covered(there):
            if have and there.get("vort_hours", 0) >= have.get("vort_hours", 0):
                remove.append(key)        # the archive copy is at least as complete
            continue
        if archive is None and not have:
            continue                      # cannot tell what is archived: do not start a backlog
        if not have or have.get("error") or have.get("t_end") != e.get("t_end"):
            ended.append(key)             # new, failed before, or its window has changed
        elif have.get("era5_pending"):
            old = (now - datetime.fromtimestamp(e["t_end"], UTC)) >= timedelta(days=config.ERA5_MIN_AGE_DAYS)
            due = now.timestamp() - (have.get("era5_try") or 0) >= config.ERA5_RETRY_H * 3600
            if old and due:
                ended.append(key)
    return active + ended[:config.ENV_LIVE_ENDED_PER_RUN], remove


def drop(key, index):
    """Delete a record's files from the live folder, and its stored frames."""
    index["cases"].pop(key, None)
    folder = config.ENV_DIR
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name == key + ".json" or (name.startswith(key + "_") and name.endswith(".webp")):
                os.remove(os.path.join(folder, name))
    if os.path.exists(prior_path(key)):
        os.remove(prior_path(key))


def run(now=None, archive=None, max_minutes=None, say=log):
    began = time.monotonic()
    now = now or datetime.now(UTC)
    storms = process.load_index()["storms"]
    index = run_env.load_index()
    todo, remove = plan(storms, index["cases"], archive, now)
    for key in remove:
        drop(key, index)
        say(f"  {key}: removed (archived or no longer listed)")
    for key in todo:
        if time.monotonic() - began > (max_minutes or config.ENV_LIVE_MAX_MIN) * 60:
            say("  out of time; the rest is left for the next update")
            break
        path = os.path.join(config.DATA_DIR, "storms", key + ".json")
        try:
            with open(path) as fh:
                product = json.load(fh)
            say(f"{key} {product['label']}")
            summary, prior = env.build_live(product, load_prior(key), log=say, now=now)
        except Exception as err:
            say(f"{key}: FAILED {type(err).__name__}: {err}")
            have = index["cases"].get(key)
            if not have:                  # keep whatever was published before; note the failure only if there is none
                index["cases"][key] = {"error": f"{type(err).__name__}: {err}"[:300], "frames": 0, "hours": 0, "vort_hours": 0, "updated": now.strftime("%Y-%m-%dT%H:%MZ")}
            continue
        index["cases"][key] = summary
        if summary["frames"]:
            save_prior(key, prior)
        run_env.save_index(index, now)
    run_env.save_index(index, now)
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--now", help="pretend it is this UTC time (ISO), for testing and reruns")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args(argv)
    now = datetime.fromisoformat(args.now).replace(tzinfo=UTC) if args.now else datetime.now(UTC)
    log(f"infrared update for {now:%Y-%m-%d %H:%MZ}")
    run(now=now, archive=archive_index(), max_minutes=args.max_minutes)
    env._close_pools()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

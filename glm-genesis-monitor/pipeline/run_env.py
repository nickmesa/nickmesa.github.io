"""Add channel 13 IR and ERA5 vorticity to cases that are already in the archive.

    python -m pipeline.run_env --id al092022,al022024
    python -m pipeline.run_env --year 2022 --basin al

Reads the published records in <data>/storms/ (for the window and the
interpolated centres) and writes to config.ENV_DIR. <env>/index.json lists what
has been done; a case in it is skipped unless --redo is given, or --fill is
given and it is still missing ERA5 hours (NCAR publishes ERA5 some months
behind real time).
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

from . import config, env

UTC = timezone.utc


def log(msg):
    print(msg, flush=True)


def index_path(folder=None):
    return os.path.join(folder or config.ENV_DIR, "index.json")


def load_index(folder=None):
    path = index_path(folder)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"updated": None, "cases": {}}


def save_index(index, now=None, folder=None):
    index["updated"] = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%MZ")
    index["cases"] = dict(sorted(index["cases"].items()))
    path = index_path(folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(index, fh, separators=(",", ":"))
    os.replace(path + ".tmp", path)


def archived(year=None, basin=None):
    """Ids of finished records in the lightning archive, optionally for one season and basin."""
    with open(os.path.join(config.DATA_DIR, "index.json")) as fh:
        storms = json.load(fh)["storms"]
    return [e["id"] for e in storms
            if e["status"] == "ended" and (year is None or e["year"] == year) and (basin is None or e["basin"].lower() == basin)]


def run(ids, redo=False, fill=False, max_hours=5.0, now=None):
    began = time.monotonic()
    index = load_index()
    for key in ids:
        have = index["cases"].get(key)
        if have and not redo and not have.get("error") and not (fill and have["vort_hours"] < have["hours"]):
            continue
        if time.monotonic() - began > max_hours * 3600:
            log(f"stopping after {max_hours} h; run again to finish the rest")
            break
        path = os.path.join(config.DATA_DIR, "storms", key + ".json")
        try:
            with open(path) as fh:
                product = json.load(fh)
            log(f"{key} {product['label']}")
            summary = env.build(product, log=log, now=now)
        except Exception as err:
            summary = {"error": f"{type(err).__name__}: {err}"[:300], "frames": 0, "hours": 0, "vort_hours": 0}
            log(f"{key}: FAILED {summary['error']}")
        index["cases"][key] = summary
        save_index(index, now)
    return index


def merge(incoming, now=None, log=log):
    """Add the output folders of parallel jobs to config.ENV_DIR (the collect step)."""
    index = load_index()
    added = 0
    for root, _, names in os.walk(incoming):
        if "index.json" not in names:
            continue
        theirs = load_index(root)["cases"]
        for key, summary in theirs.items():
            mine = index["cases"].get(key)
            if mine and mine.get("updated", "") >= summary.get("updated", "") and not mine.get("error"):
                continue
            if not os.path.exists(os.path.join(root, key + ".json")):
                if not mine:
                    index["cases"][key] = summary          # a failure: keep the reason on record
                continue
            for name in os.listdir(config.ENV_DIR) if os.path.isdir(config.ENV_DIR) else []:
                if name.startswith(key + "_") and name.endswith(".webp"):
                    os.remove(os.path.join(config.ENV_DIR, name))
            os.makedirs(config.ENV_DIR, exist_ok=True)
            for name in names:
                if name == key + ".json" or (name.startswith(key + "_") and name.endswith(".webp")):
                    shutil.copy2(os.path.join(root, name), os.path.join(config.ENV_DIR, name))
            index["cases"][key] = summary
            added += 1
    save_index(index, now)
    log(f"{added} cases added; {len(index['cases'])} in the index")
    return added


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", help="record id(s), comma separated")
    ap.add_argument("--year", type=int)
    ap.add_argument("--basin", choices=config.BASINS)
    ap.add_argument("--redo", action="store_true", help="process cases again that are already done")
    ap.add_argument("--fill", action="store_true", help="process cases again that are missing ERA5 hours")
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--merge", help="add the output folders found under this folder, then stop")
    args = ap.parse_args(argv)
    if args.merge:
        merge(args.merge)
        return 0
    ids = [s.strip() for s in args.id.split(",") if s.strip()] if args.id else archived(args.year, args.basin)
    index = run(ids, redo=args.redo, fill=args.fill, max_hours=args.max_hours)
    failed = [k for k in ids if index["cases"].get(k, {}).get("error")]
    if failed:
        log(f"with errors: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

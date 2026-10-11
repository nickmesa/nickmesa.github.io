"""Add finished records produced elsewhere to the data folder.

    python -m pipeline.collect <folder>

A season backfill runs as several parallel jobs that each write records to
their own folder. This step, the only one that touches the data branch, adds
those records and the season plans to it and rebuilds the index entries.
"""
import json
import os
import sys
from datetime import datetime, timezone

from . import config, process


def collect(folder, now=None, log=print):
    now = now or datetime.now(timezone.utc)
    index = process.load_index()
    added = plans = 0
    for root, _, names in os.walk(folder):
        for name in sorted(names):
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path) as fh:
                    obj = json.load(fh)
            except ValueError:
                log(f"  {path}: not readable, skipped")
                continue
            if not isinstance(obj, dict):
                continue
            if "features" in obj and "steps" in obj and "id" in obj:
                entry = process.publish(obj, index)
                added += 1
                log(f"  {entry['id']} {entry['label']}: {entry['n_features']} features")
            elif "cases" in obj and "year" in obj and "basin" in obj:
                process._write_json(os.path.join(config.DATA_DIR, "backfill", f"{obj['year']}-{obj['basin']}.json"), obj)
                plans += 1
    if added:
        process.save_index(index, now)
    log(f"{added} records and {plans} season plans added")
    return added, plans


if __name__ == "__main__":
    collect(sys.argv[1])

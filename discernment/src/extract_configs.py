"""Extract every life's launch settings from the experiment databases into data/configs.csv.

Each life stored the arguments it was started with (or, for a life that was resumed or extended, last started
with) in the `config` column of its database's `runs` table. This script reads them straight from the SQLite
files, keeps the same copy of each life as src/extract_data.py (the one with most progress, never a stopped one),
and writes one row per life of data/lives.csv:

  python src/extract_configs.py '<dir with the experiment databases>/*.db' data

A setting missing from a life's config means the life ran before that option existed, with its default
(WebShop lives: lesson_once false, min_lessons 0, user "templates" = the plain user, brain "constant" = the plain
adapter). τ²-bench lives (lab.life_tau2) have none of these options: each lesson always counts once there.
Machine-local values (server URLs and ports, absolute paths, error messages) are dropped or made relative.
`hours` is wall-clock time from first start to finish, so a life that was paused and resumed counts its pause.
"""

import csv
import glob
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DROP = {"backend", "vllm_url", "sense_url", "proxy_port", "deadline_hours", "error", "resume", "name", "experiment"}
DEFAULTS = {"lesson_once": False, "min_lessons": 0, "user": "templates", "brain": "constant"}  # WebShop (lab.life) only
FIRST = ["life", "paper", "lab_experiment", "judge", "seed", "model", "days", "started_utc", "finished_utc", "hours"]


def local(value):
    """Make a path from the experiment VMs relative to the code folder."""
    if isinstance(value, str) and "/alife/" in value:
        return value.split("/alife/", 1)[1]
    return value


def main(pattern: str, out: Path):
    lives = {r["life"]: r for r in csv.DictReader(open(out / "lives.csv"))}
    best = {}
    for path in sorted(glob.glob(pattern)):
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for name, status, done, started, finished, config in con.execute(
                "select name, status, done, started, finished, config from runs"):
            if name in lives and status != "stopped" and (name not in best or (done or 0) > (best[name][1] or 0)):
                best[name] = (status, done, started, finished, json.loads(config or "{}"))
        con.close()
    missing = sorted(set(lives) - set(best))
    if missing:
        sys.exit(f"no database row for {missing}")

    rows = []
    for name in sorted(lives):
        status, done, started, finished, config = best[name]
        life = lives[name]
        row = {"life": name, "paper": life["paper"], "lab_experiment": life["lab_experiment"], "judge": life["judge"],
               "seed": life["seed"], "model": life["model"], "days": config.get("days", ""),
               "started_utc": stamp(started), "finished_utc": stamp(finished),
               "hours": f"{(finished - started) / 3600:.1f}" if started and finished else ""}
        base = {} if life["lab_experiment"].startswith("tau2") else DEFAULTS
        settings = {**base, **{k: local(v) for k, v in config.items() if k not in DROP}}
        for k in ("model", "seed", "days"):
            settings.pop(k, None)
        row.update({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in settings.items()})
        rows.append(row)

    keys = FIRST + sorted({k for r in rows for k in r} - set(FIRST))
    with open(out / "configs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{out / 'configs.csv'}: {len(rows)} rows, {len(keys)} columns")


def stamp(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M") if t else ""


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))

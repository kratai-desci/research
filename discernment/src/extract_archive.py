"""Build the level-3 archive: every step of every attempt of the paper's 127 lives, in one SQLite file.

  python src/extract_archive.py '<dir with the experiment databases>/*.db' data/lives.csv <out.db>

The experiment databases hold one file per machine, with other experiments and stale copies of lives that were
moved between machines. This keeps, for each life of lives.csv, the copy with most progress (as extract_data.py
does) and copies its rows from four tables:

  runs      one row per life; `config` holds its launch settings (machine-local values removed, as in configs.csv)
  episodes  one row per attempt: practice, retry or exam (split), day, task, the user's reaction (message), score
  steps     one row per step of an attempt: the model's thought, action and input, and what it saw back
  checks    per-attempt checks: the judge's verdict and, on exam days, the sense's readings of fixed reactions

Ids are renumbered so the file is self-contained; run_id and episode_id keep the links between tables.
"""

import csv
import glob
import json
import sqlite3
import sys
from pathlib import Path

DROP = {"backend", "vllm_url", "sense_url", "proxy_port", "deadline_hours", "error", "resume"}


def clean(config: str) -> str:
    d = json.loads(config or "{}")
    d = {k: (v.split("/alife/", 1)[1] if isinstance(v, str) and "/alife/" in v else v)
         for k, v in d.items() if k not in DROP}
    return json.dumps(d, sort_keys=True)


def main(pattern: str, lives_csv: str, out: str):
    wanted = {r["life"] for r in csv.DictReader(open(lives_csv))}
    best = {}
    for path in sorted(glob.glob(pattern)):
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for rid, name, status, done in con.execute("select id, name, status, done from runs"):
            if name in wanted and status != "stopped" and (name not in best or (done or 0) > (best[name][2] or 0)):
                best[name] = (path, rid, done)
        con.close()
    assert set(best) == wanted, sorted(wanted - set(best))

    Path(out).unlink(missing_ok=True)
    dst = sqlite3.connect(out)
    src0 = sqlite3.connect(f"file:{best[min(best)][0]}?mode=ro", uri=True)
    for table in ("runs", "episodes", "steps", "checks"):
        dst.execute(src0.execute("select sql from sqlite_master where name=?", (table,)).fetchone()[0])
    src0.close()

    by_file = {}
    for name, (path, rid, _) in best.items():
        by_file.setdefault(path, []).append((name, rid))
    run_id = episode_id = 0
    for path, runs in sorted(by_file.items()):
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for name, rid in sorted(runs):
            run_id += 1
            row = src.execute("select * from runs where id=?", (rid,)).fetchone()
            cols = [c[1] for c in src.execute("pragma table_info(runs)")]
            row = dict(zip(cols, row), id=run_id)
            row["config"] = clean(row["config"])
            dst.execute(f"insert into runs ({','.join(row)}) values ({','.join('?' * len(row))})", list(row.values()))
            ecols = [c[1] for c in src.execute("pragma table_info(episodes)")]
            scols = [c[1] for c in src.execute("pragma table_info(steps)")]
            ccols = [c[1] for c in src.execute("pragma table_info(checks)")]
            for ep in src.execute("select * from episodes where run_id=? order by id", (rid,)).fetchall():
                episode_id += 1
                e = dict(zip(ecols, ep))
                old = e["id"]
                e.update(id=episode_id, run_id=run_id)
                dst.execute(f"insert into episodes ({','.join(e)}) values ({','.join('?' * len(e))})", list(e.values()))
                for st in src.execute("select * from steps where episode_id=? order by n", (old,)):
                    s = dict(zip(scols, st), episode_id=episode_id)
                    s.pop("id")
                    dst.execute(f"insert into steps ({','.join(s)}) values ({','.join('?' * len(s))})", list(s.values()))
                for ch in src.execute("select * from checks where episode_id=? order by n", (old,)):
                    c = dict(zip(ccols, ch), episode_id=episode_id)
                    c.pop("id")
                    dst.execute(f"insert into checks ({','.join(c)}) values ({','.join('?' * len(c))})", list(c.values()))
            print(f"{name}: run {run_id}, episodes up to {episode_id}", flush=True)
        src.close()
        dst.commit()
    dst.execute("create index episodes_by_run on episodes(run_id, split, day)")
    dst.execute("create index steps_by_episode on steps(episode_id)")
    dst.execute("create index checks_by_episode on checks(episode_id, kind)")
    dst.commit()
    dst.execute("vacuum")
    dst.close()


if __name__ == "__main__":
    main(*sys.argv[1:4])

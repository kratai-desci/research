"""Extract the paper's level-2 data from the experiment databases into data/.

Run from a checkout of the code the experiments ran on (the alife repository; see README for the commit):
  LAB_DB_SOURCE='<dir with the experiment databases>/*.db' uv run python src/extract_data.py <data dir>

Each life (one seed) is one row of lives.csv; every exam score, every practice attempt and every lesson is kept,
so every number, test and figure in the paper can be recomputed from data/ alone (src/stats.py, src/figures.py).
"""

import csv
import json
import gzip
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ALIFE_REPO", "."))
from lab import api  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

# lab experiment name -> where it appears in the paper
PAPER = {
    "growing-from-use": "Exp 1 pilot",
    "growing-from-use-14d": "Exp 1",
    "lifespan-365d": "Exp 1 year",
    "tree-rings-14d": "Exp 1 adapters",
    "feedback-sense-14d": "Exp 2",
    "realistic-user-14d": "Exp 3",
    "lesson-once-14d": "Exp 3 sleep variant",
    "clear-verdict-14d": "Exp 4",
    "tau2-learning-30d": "Exp 5",
}
JUDGE = {"user": "stated verdict"}  # the paper's name for the lab's "user" judge


def lives() -> list[dict]:
    """One copy per life (a VM cloned from another's disk also carries stale copies): the one with most progress."""
    best = {}
    for r in api.all_runs():
        if r["exp"] in PAPER and r["status"] != "stopped":
            if r["name"] not in best or (r["done"] or 0) > (best[r["name"]]["done"] or 0):
                best[r["name"]] = r
    return sorted(best.values(), key=lambda r: r["name"])


def write(name: str, rows: list[dict], gz: bool = False):
    path = OUT / (name + (".csv.gz" if gz else ".csv"))
    opener = (lambda p: gzip.open(p, "wt", newline="", encoding="utf-8")) if gz else (lambda p: open(p, "w", newline="", encoding="utf-8"))
    with opener(path) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


runs = lives()
ids = [r["id"] for r in runs]
key = {r["id"]: r["name"] for r in runs}
tau2 = {r["id"] for r in runs if r["exp"] == "tau2-learning-30d"}

write("lives", [{
    "life": r["name"], "paper": PAPER[r["exp"]], "lab_experiment": r["exp"],
    "judge": JUDGE.get(r["condition"], r["condition"]), "seed": (re.search(r"-s(\d+)$", r["name"]) or [None, ""])[1],
    "model": r["model"], "status": r["status"],
} for r in runs])

# exams: the mean score of each life on each day and exam split (tau2: also per domain)
rows = api.by_run(ids, """SELECT run_id, day, split, task, COUNT(*) n, AVG(score) mean_score,
                         SUM(score >= 100) full_success FROM episodes WHERE run_id IN ({runs}) AND split LIKE 'probe%'
                         AND status = 'done' GROUP BY run_id, day, split, task""")
exam = {}
for x in rows:  # WebShop exams are keyed by goal, so fold tasks together there; keep tau2's domain
    dom = x["task"] if x["run_id"] in tau2 else ""
    k = (x["run_id"], x["day"], x["split"], dom)
    e = exam.setdefault(k, {"n": 0, "sum": 0.0, "full": 0})
    e["n"] += x["n"]; e["sum"] += x["mean_score"] * x["n"]; e["full"] += x["full_success"] or 0
errors = {(x["run_id"], x["day"], x["split"], x["task"] if x["run_id"] in tau2 else ""): x["n"] for x in api.by_run(
    ids, "SELECT run_id, day, split, task, COUNT(*) n FROM episodes WHERE run_id IN ({runs}) AND split LIKE 'probe%' "
         "AND status != 'done' GROUP BY run_id, day, split, task")}
write("exams", [{
    "life": key[k[0]], "day": k[1], "split": k[2], "domain": k[3], "tasks": e["n"], "errors": errors.get(k, 0),
    "mean_score": round(e["sum"] / e["n"], 6), "full_successes": e["full"],
} for k, e in sorted(exam.items(), key=lambda kv: (key[kv[0][0]], kv[0][1], kv[0][2], kv[0][3]))])

# every practice attempt and retry, with the judge's verdict and the user's reaction
att = api.by_run(ids, """SELECT e.run_id, e.id, e.day, e.split, e.task, e.slot, e.status, e.score, e.message, e.answer,
       (SELECT c.evidence FROM checks c WHERE c.episode_id = e.id AND c.name LIKE 'Creature''s verdict%' LIMIT 1) verdict,
       (SELECT c.evidence FROM checks c WHERE c.episode_id = e.id AND c.name LIKE 'Verdict (%' LIMIT 1) tau2_verdict,
       (SELECT c.evidence FROM checks c WHERE c.episode_id = e.id AND c.name = 'User said' LIMIT 1) said
       FROM episodes e WHERE e.run_id IN ({runs}) AND e.split IN ('practice', 'retry')""")


def lesson(a) -> str:
    """yes / no / none: whether the judge made this attempt a lesson."""
    if a["run_id"] in tau2:
        v = a["tau2_verdict"] or ""
        if v.startswith("reward"):
            m = re.match(r"reward ([\d.]+|None)", v)
            return "none" if not m or m[1] == "None" else ("yes" if float(m[1]) >= 1 else "no")
        m = re.match(r"sense (\w+)", v)
        return "none" if not m or m[1] == "null" else ("yes" if m[1] in ("80", "100") else "no")
    v = a["verdict"] or ""
    return "yes" if v.startswith("good") else "no" if v.startswith("bad") else "none"


def ending(answer):
    """tau2: what the sense read, stored with the conversation (the customer's request and last words)."""
    try:
        return json.loads(answer or "{}")
    except json.JSONDecodeError:
        return {"said": ""}


out = []
for a in att:
    out.append({
        "life": key[a["run_id"]], "day": a["day"], "kind": a["split"], "task": a["slot"] if a["run_id"] not in tau2 else f"{a['task']}/{a['slot']}",
        "status": a["status"], "score": a["score"], "lesson": lesson(a),
        "verdict": (a["tau2_verdict"] if a["run_id"] in tau2 else a["verdict"]) or "",
        "request": (a["message"] or "")[:300],
        "user_said": ending(a["answer"])["said"] if a["run_id"] in tau2 else (a["said"] or ""),
    })
out.sort(key=lambda r: (r["life"], r["day"], r["kind"], str(r["task"])))
write("attempts", out, gz=True)

# the sense's daily accuracy checks on a fixed set of recorded reactions (Experiment 2: "82.6% on day 0, 82.8% on day 14")
chk = api.by_run(ids, """SELECT run_id, day, task, slot, score, message, answer FROM episodes
                        WHERE run_id IN ({runs}) AND split = 'sense-check' AND status = 'done'""")
if chk:
    write("sense_checks", [{"life": key[c["run_id"]], "day": c["day"], "kind": c["task"], "item": c["slot"],
                            "correct": int((c["score"] or 0) >= 100), "said": c["message"], "sense_read": c["answer"]}
                           for c in sorted(chk, key=lambda c: (key[c["run_id"]], c["day"], str(c["slot"])))], gz=True)

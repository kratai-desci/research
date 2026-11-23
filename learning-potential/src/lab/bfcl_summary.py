"""E11: summarise a bfcl-eval project folder into one JSON for the dashboard.

    python lab/bfcl_summary.py <BFCL_PROJECT_ROOT> <stage> <message>  > e11.json

Reads, for each model, every per-category score file (score/<model>/**/BFCL_*_<category>_score.json, whose
first line holds accuracy, correct_count and total_count), the leaderboard tables bfcl writes
(score/data_*.csv), and the same from score-repeat/ (the stability rerun of multi-turn). Only the standard
library, so it runs in any venv.

Progress: per model and category, entries generated so far (lines in result/<model>/**/BFCL_v4_<cat>_result.json)
out of the dataset's entries (bfcl_eval/data); and per bfcl generate call (one per model and category group), the
last progress bar in logs/bfcl-generate.log, from which the time left is estimated.
"""
import csv
import json
import os
import re
import sys
import time
from pathlib import Path


def categories(score: Path) -> dict:
    out = {}
    for f in sorted(score.glob("*/**/*_score.json")):
        model = f.relative_to(score).parts[0]
        m = re.match(r"BFCL_v\d+_(.+)_score\.json", f.name)
        if not m:
            continue
        try:
            head = json.loads(f.read_text().splitlines()[0])
        except (IndexError, json.JSONDecodeError):
            continue
        out.setdefault(model, {})[m.group(1)] = {k: head.get(k) for k in ("accuracy", "correct_count", "total_count")}
    return out


GROUPS = {"single_turn": ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
                          "irrelevance", "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
                          "live_irrelevance", "live_relevance"],
          "multi_turn": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
          "memory": ["memory_kv", "memory_vector", "memory_rec_sum"]}
SECONDS_PER_ENTRY = {"single_turn": 0.25, "multi_turn": 11.0, "memory": 5.0}  # guesses until a run has been timed
BAR = re.compile(r"^Generating results for (\S+?):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+)")


def lines(f: Path) -> int:
    try:
        return sum(1 for line in f.open() if line.strip())
    except OSError:
        return 0


def data_totals() -> dict:
    import importlib.util
    spec = importlib.util.find_spec("bfcl_eval")
    if not spec or not spec.origin:
        return {}
    data = Path(spec.origin).parent / "data"
    return {c: lines(data / f"BFCL_v4_{'memory' if c.startswith('memory') else c}.json")
            for cs in GROUPS.values() for c in cs}


def generated(result: Path, models: list[str], totals: dict) -> dict:
    """{model folder: {category: [done, total]}} for every planned model and category."""
    out = {}
    for m in models:
        folder = m.replace("/", "_")
        files = {re.sub(r"^BFCL_v\d+_|_result\.json$", "", f.name): f for f in (result / folder).glob("**/*_result.json")}
        out[folder] = {c: [lines(files[c]) if c in files else 0, totals.get(c)]
                       for g in GROUP_ORDER for c in GROUPS.get(g, [g])}
    return out


def secs(t: str) -> float | None:
    if "?" in t:
        return None
    n = 0.0
    for part in t.split(":"):
        n = n * 60 + float(part)
    return n


def bars(log: Path) -> list[dict]:
    """One entry per bfcl generate call: the last progress bar it printed."""
    runs = []
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return runs
    for line in re.split(r"[\r\n]", text):
        m = BAR.match(line)
        if not m:
            continue
        model, done, total, elapsed, left = m.group(1), int(m.group(2)), int(m.group(3)), secs(m.group(4)), secs(m.group(5))
        last = runs[-1] if runs else None
        if last and last["model"] == model and last["total"] == total and done >= last["done"]:
            last.update(done=done, elapsed=elapsed, left=left)
        else:
            runs.append({"model": model, "done": done, "total": total, "elapsed": elapsed, "left": left})
    return runs


def plan(runs: list[dict], models: list[str], stage: str, totals: dict) -> tuple[list[dict], float | None]:
    """Every planned generate call (model x group, then the multi-turn repeat), with its state and time left."""
    planned = [(m, g) for m in models for g in GROUP_ORDER]
    if not os.environ.get("NO_REPEAT"):
        planned.append((models[-1], "repeat"))
    seen: dict[str, int] = {}
    for r in runs:  # the k-th bar of a model is its k-th group
        k = seen.get(r["model"], 0)
        seen[r["model"]] = k + 1
        order = GROUP_ORDER + ["repeat"]
        r["group"] = order[k] if k < len(order) else "?"
    speed = {}  # (model, group) -> seconds per entry, from finished calls
    for r in runs:
        if r["done"] == r["total"] and r["elapsed"]:
            g = "multi_turn" if r["group"] == "repeat" else r["group"]
            speed[(r["model"], g)] = r["elapsed"] / r["total"]
    out, left_all = [], 0.0
    for m, g in planned:
        base = "multi_turn" if g == "repeat" else g
        r = next((x for x in runs if x["model"] == m and x["group"] == g), None)
        total = r["total"] if r else sum(totals.get(c) or 0 for c in GROUPS.get(base, [base])) or None
        if r and r["done"] == r["total"]:
            out.append({"model": m, "group": g, "state": "done", "done": r["done"], "total": r["total"], "left": 0})
            continue
        if r and stage not in ("done", "failed"):
            left = r["left"]
            state = "running"
        else:
            spe = speed.get((m, base)) or next((v for (mm, gg), v in speed.items() if gg == base), None) \
                  or SECONDS_PER_ENTRY.get(base, 5.0)
            left = spe * (total or 0) * (1.3 if "1.7B" in m and (m, base) not in speed else 1.0)
            state = "planned"
        out.append({"model": m, "group": g, "state": state, "done": r["done"] if r else 0, "total": total,
                    "left": left, "elapsed": r["elapsed"] if r else None})
        left_all = None if left is None or left_all is None else left_all + left
    return out, left_all


GROUP_ORDER = os.environ.get("CATEGORIES", "single_turn multi_turn memory").split()


def tables(score: Path) -> dict:
    out = {}
    for f in sorted(score.glob("data_*.csv")):
        with f.open() as fh:
            out[f.stem] = list(csv.DictReader(fh))
    return out


root = Path(sys.argv[1])
try:  # keep the first start time across updates
    started = json.loads((root / "e11.json").read_text())["started"]
except (OSError, KeyError, json.JSONDecodeError):
    started = time.time()
summary = {"stage": sys.argv[2], "message": sys.argv[3], "started": started, "updated": time.time(),
           "bfcl_version": os.environ.get("BFCL_VERSION")}
score = root / ("score-run1" if sys.argv[2] == "repeating" else "score")  # run_bfcl.sh parks run 1 there meanwhile
early = Path.home() / "bfcl-early" / "score"  # scored so far by deploy/gcp/bfcl_early.sh, while generation runs
if sys.argv[2] not in ("done", "failed", "repeating") and early.exists():
    score, summary["partial"] = early, True
if score.exists():
    summary["categories"] = categories(score)
    summary["tables"] = tables(score)
if (root / "score-repeat").exists():
    summary["repeat"] = categories(root / "score-repeat")
models = os.environ.get("MODELS", "Qwen/Qwen3-0.6B-FC Qwen/Qwen3-1.7B-FC").split()
totals = data_totals()
result = root / ("result-run1" if sys.argv[2] == "repeating" else "result")
summary["generated"] = generated(result, models, totals)
log = Path(os.environ.get("BFCL_LOG", Path.home() / "alife" / "logs" / "bfcl-generate.log"))
summary["calls"], summary["time_left"] = plan(bars(log), models, sys.argv[2], totals)
print(json.dumps(summary))

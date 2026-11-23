"""E14: live progress of one model's bfcl run, for the dashboard: per category, answers generated so far out of the
dataset's entries, by counting lines in result/**/BFCL_v4_<category>_result.json.

    python -m lab.bfcl_progress <BFCL_PROJECT_ROOT> <name>  > progress.json

Runs in bfcl-eval's venv (the dataset sizes come from bfcl_eval/data). The categories are those of CATEGORIES
(groups, as run_bfcl_arms.sh), single_turn multi_turn memory by default.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

# as lab/bfcl_summary.py (which runs as a script on import, so it is not imported here)
GROUPS = {"single_turn": ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
                          "irrelevance", "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
                          "live_irrelevance", "live_relevance"],
          "multi_turn": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
          "memory": ["memory_kv", "memory_vector", "memory_rec_sum"]}


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


def progress(root: Path, name: str) -> dict:
    totals = data_totals()
    cats = [c for g in os.environ.get("CATEGORIES", "single_turn multi_turn memory").split() for c in GROUPS.get(g, [g])]
    done = {}
    for f in (root / "result").glob("**/*_result.json"):
        cat = re.sub(r"^BFCL_v\d+_|_result\.json$", "", f.name)
        if cat in cats:
            done[cat] = lines(f)
    per = {c: [min(done.get(c, 0), totals.get(c) or 0), totals.get(c)] for c in cats}
    return {"name": name, "updated": time.time(), "categories": per,
            "done": sum(d for d, _ in per.values()), "total": sum(t or 0 for _, t in per.values())}


if __name__ == "__main__":
    print(json.dumps(progress(Path(sys.argv[1]), sys.argv[2])))

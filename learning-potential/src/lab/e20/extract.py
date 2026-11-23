"""E20: the per-question results behind every E14 to E19 claim, extracted once into small files kept in the repo.

Three tables, each a fixed task order plus one string of 0s and 1s per run (1 = solved, "-" = not attempted):

  bfcl.json      every BFCL v4 question (each category), for each model's E14 floor and every E15 to E19 exam.
                 BFCL's score files list only the failed questions, so solved = answered (result file) minus failed
                 (score file); each category's count is checked against the score file's own correct_count.
  probe.json     the 200 probe tasks of data/split.json, for each model's base (checkpoint 0) and every probe
                 snapshot (every 100 tasks) of every E15, E16, E17 and E19 run.
  practice.json  the 1,200 practice tasks of the split (seed 15's order), for E15's frozen runs and every learning
                 run; runs with another seed did the same tasks in another order and are aligned by task.

    python -m lab.e20.extract --runs <folder with e15/ e16/ e17/ e19/ as in the bucket> --exams <bfcl/e18 folder> \
        --floors <bfcl/e14floor-work folder> --e11 <bfcl/e11 folder> --out ../data/per_question
"""
import argparse
import json
import re
from pathlib import Path

SPLIT = Path(__file__).resolve().parents[3] / "data" / "split.json"
MODELS = ["qwen3-1.7b", "lfm2.5-1.2b", "qwen3.5-0.8b", "llama3.2-1b", "falcon3-1b", "gemma3-1b"]


def natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def lines(p: Path) -> list[dict]:
    out = []
    for l in p.read_text().splitlines():
        l = l.strip()
        if l:
            out.append(json.loads(l))
    return out


def bfcl_exam(score_root: Path, result_root: Path) -> dict[str, dict[str, int]]:
    """{category: {question id: 0/1}} for one exam folder (one model inside it)."""
    out = {}
    for sf in sorted(score_root.rglob("BFCL_v4_*_score.json")):
        cat = sf.name[len("BFCL_v4_"):-len("_score.json")]
        rel = sf.relative_to(score_root)
        rf = result_root / rel.parent / sf.name.replace("_score.json", "_result.json")
        if not rf.exists():
            continue
        ls = lines(sf)
        head, failed = ls[0], {x["id"] for x in ls[1:] if "id" in x}
        ids = [x["id"] for x in lines(rf) if "id" in x]
        if len(ids) != head.get("total_count"):
            raise SystemExit(f"{sf}: {len(ids)} answers, score file says {head.get('total_count')}")
        got = {i: int(i not in failed) for i in ids}
        if sum(got.values()) != head.get("correct_count"):
            raise SystemExit(f"{sf}: {sum(got.values())} solved, score file says {head.get('correct_count')}")
        out[cat] = got
    return out


def one_model(folder: Path, model_dir: str | None = None) -> tuple[Path, Path]:
    """The score and result folders of the (single, or named) model in an exam folder."""
    score, result = folder / "score", folder / "result"
    if model_dir is None:
        dirs = [d for d in score.iterdir() if d.is_dir()]
        if len(dirs) != 1:
            raise SystemExit(f"{folder}: expected one model, found {[d.name for d in dirs]}")
        model_dir = dirs[0].name
    return score / model_dir, result / model_dir


def bits(order: list, got: dict) -> str:
    return "".join("-" if k not in got else str(got[k]) for k in order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--exams", required=True)
    ap.add_argument("--floors", required=True)
    ap.add_argument("--e11", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, exams, floors = Path(a.runs), Path(a.exams), Path(a.floors)

    # ---- BFCL ----
    table = {}
    for key in MODELS:  # the floors: E14 (Qwen3-1.7B's is E11's exam, as on the dashboard)
        if key == "qwen3-1.7b":
            table["floor-qwen3-1.7b"] = bfcl_exam(*one_model(Path(a.e11), "Qwen_Qwen3-1.7B-FC"))
        else:
            table[f"floor-{key}"] = bfcl_exam(*one_model(floors / key / "e14floor" / key))  # not e14floor-v3 (E13's pool)
    for d in sorted(p for p in exams.iterdir() if p.is_dir() and (p / "score").is_dir()):
        summary = exams / f"{d.name}.json"
        if not summary.exists() or json.loads(summary.read_text()).get("stage") != "done":
            continue  # an unfinished exam is not a result
        table[d.name] = bfcl_exam(*one_model(d))
    cats = {}
    for t in table.values():
        for c, got in t.items():
            cats.setdefault(c, set()).update(got)
    order = {c: sorted(ids, key=natural) for c, ids in sorted(cats.items())}
    (out / "bfcl.json").write_text(json.dumps({
        "about": "BFCL v4, every question: 1 solved, 0 not, - not in that exam. Order: 'questions' per category.",
        "questions": order,
        "exams": {name: {c: bits(order[c], got) for c, got in sorted(t.items())} for name, t in sorted(table.items())},
    }, indent=1) + "\n")

    # ---- probe and practice ----
    sp = json.loads(SPLIT.read_text())
    probe_order = [tuple(k) for k in sp["probe"]]
    prac_order = [tuple(k) for k in sp["practice"]]
    probes, practice = {}, {}
    for key in MODELS:
        p0 = runs / "e15" / key / "probe-0.json"
        if p0.exists():
            probes[f"base/{key}"] = bits(probe_order, {(r["set"], r["id"]): int(r["valid"])
                                                       for r in json.loads(p0.read_text())["results"]})
    for exp in ("e15", "e16", "e17", "e19"):
        for key in MODELS:
            for d in sorted((runs / exp / key).glob("*")) if (runs / exp / key).is_dir() else []:
                if not d.is_dir() or d.name.startswith(("pilot", "check")):
                    continue
                st = d / "status.json"
                if not st.exists() or json.loads(st.read_text()).get("stage") != "done":
                    continue  # unfinished (e.g. Qwen3-1.7B's stopped extra seeds of E15 and E16)
                log = d / "log.jsonl"
                if log.exists():
                    got = {}
                    for r in lines(log):
                        got.setdefault((r["set"], r["id"]), int(r["valid"]))
                    practice[f"{exp}/{key}/{d.name}"] = bits(prac_order, got)
                for f in sorted(d.glob("probe-*.json"), key=lambda f: int(f.stem.split("-")[1])):
                    res = json.loads(f.read_text()).get("results")
                    if res:
                        probes[f"{exp}/{key}/{d.name}/{f.stem.split('-')[1]}"] = bits(
                            probe_order, {(r["set"], r["id"]): int(r["valid"]) for r in res})
    (out / "probe.json").write_text(json.dumps({
        "about": "The 200 probe tasks: 1 solved, 0 not. base/<model> is the model with nothing learnt (checkpoint 0); "
                 "<exp>/<model>/<run>/<n> is the run's state after n practice tasks.",
        "tasks": [list(k) for k in probe_order], "runs": probes}, indent=1) + "\n")
    (out / "practice.json").write_text(json.dumps({
        "about": "The 1,200 practice tasks in seed 15's order: 1 solved, 0 not, - not done. e15/<model>/frozen-s15 is "
                 "the frozen model (its first 1,200 tasks are these).",
        "tasks": [list(k) for k in prac_order], "runs": practice}, indent=1) + "\n")
    print(json.dumps({"bfcl_exams": len(table), "categories": len(order), "probe_rows": len(probes),
                      "practice_rows": len(practice)}))


if __name__ == "__main__":
    main()

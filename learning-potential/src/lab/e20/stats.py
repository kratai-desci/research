"""E20: paired statistics for every E15 to E19 claim, from the per-question tables (lab/e20/extract.py).

Every comparison is paired: the same questions or tasks, solved or not, under two conditions.
  - Test: McNemar's exact test (two-sided binomial test on the discordant pairs).
  - Interval: 95% paired bootstrap over questions (10,000 resamples, fixed seed) for the difference in % solved.
  - Many tests at once: Holm's correction within each family (listed in the output); a claim counts as significant
    when its Holm-adjusted p is below 0.05.
  - Three-seed cells (E19): the mean over seeds with a t interval over the three seeds (seed variation), a bootstrap
    over questions of the seed mean (question variation), and a paired sign-flip test of the seed mean per question
    (Holm over the cells' 21 tests).

Families:
  bfcl_base     every learnt exam against its model's E14 floor, BFCL v4 multi-turn base (200 questions): the target.
  bfcl_groups   the same against the floor in BFCL's other groups (non-live, live, multi-turn, memory), each group's
                questions pooled with equal weight (BFCL weights its groups' subcategories, so these differences are
                close to, not equal to, the headline ones).
  probe         every run's final state (task 1,200) against the model with nothing learnt, on the 200 probe tasks.
  practice      every learning run against E15's frozen model on the same 1,200 practice tasks.
  methods       LoRA against retrieval and retrieval against notes, same model and arm, seed 15: probe and BFCL base.
  arms          LoRA arms against each other, same model and seeds: judging against teaching (LFM2.5 and Qwen3.5, seed
                15; Qwen3-1.7B, seeds 15 to 17) and the reasoning kept against teaching (Qwen3-1.7B, seeds 15 to 17), on
                BFCL base, BFCL's other groups and the probe. Three-seed comparisons use the per-item mean over seed
                pairs, with the sign-flip test and a bootstrap interval over items.

    uv run --with numpy --with scipy python -m lab.e20.stats [--per-question ../data/per_question] [--out ../data/stats.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from scipy.stats import t as tdist

B = 10_000
RNG = np.random.default_rng(20)
GROUPS = {
    "non_live": ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
                 "irrelevance"],
    "live": ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance",
             "live_relevance"],
    "multi_turn": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
    "memory": ["memory_kv", "memory_vector", "memory_rec_sum"],
}
MODELS = ["qwen3-1.7b", "lfm2.5-1.2b", "qwen3.5-0.8b", "llama3.2-1b", "falcon3-1b", "gemma3-1b"]
LABEL = {"qwen3-1.7b": "Qwen3-1.7B", "lfm2.5-1.2b": "LFM2.5-1.2B", "qwen3.5-0.8b": "Qwen3.5-0.8B",
         "llama3.2-1b": "Llama-3.2-1B", "falcon3-1b": "Falcon3-1B", "gemma3-1b": "Gemma-3-1B"}
METHOD = {"e15": "notes", "e16": "lora", "e17": "retrieval", "e19": "lora"}


def vec(s: str) -> np.ndarray:
    return np.array([-1 if c == "-" else int(c) for c in s], dtype=np.int8)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """b against a (e.g. learnt against floor) on the items both have: % solved, difference, CI, McNemar."""
    m = (a >= 0) & (b >= 0)
    a, b = a[m].astype(float), b[m].astype(float)
    n = len(a)
    d = b - a
    gain, loss = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(gain, gain + loss, 0.5).pvalue if gain + loss else 1.0
    idx = RNG.integers(0, n, size=(B, n))
    boots = d[idx].mean(axis=1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "a": round(100 * a.mean(), 1), "b": round(100 * b.mean(), 1), "diff": round(100 * d.mean(), 1),
            "ci": [round(float(lo), 1), round(float(hi), 1)], "gained": gain, "lost": loss, "p": float(p)}


def holm(rows: list[dict]) -> None:
    """Adds p_holm and significant (p_holm < 0.05) to each row, in place. A comparison with no discordant pair (the
    two conditions solve exactly the same items, e.g. 0 against 0) cannot reject; it is marked untestable and left out
    of the family size."""
    live = [r for r in rows if r.get("gained", 1) + r.get("lost", 1) > 0]
    order = sorted(range(len(live)), key=lambda i: live[i]["p"])
    m, running = len(live), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * live[i]["p"]))
        live[i]["p_holm"] = running
    for r in rows:
        r["untestable"] = "p_holm" not in r
        r.setdefault("p_holm", 1.0)
        r["significant"] = r["p_holm"] < 0.05
        r["p"], r["p_holm"] = float(f"{r['p']:.3g}"), float(f"{r['p_holm']:.3g}")


def signflip(d: np.ndarray) -> float:
    """Two-sided paired sign-flip test of mean(d) = 0 (d: one difference per item), 10,000 random flips."""
    obs = abs(d.mean())
    flips = RNG.choice([-1.0, 1.0], size=(B, len(d)))
    return float(((np.abs((flips * d).mean(axis=1)) >= obs - 1e-12).sum() + 1) / (B + 1))


def exam_key(name: str) -> tuple[str, str, str]:
    """'e16-qwen3.5-0.8b-teach-s15' -> ('e16', 'qwen3.5-0.8b', 'teach-s15')."""
    exp, rest = name.split("-", 1)
    key = next(k for k in MODELS if rest.startswith(k + "-"))
    return exp, key, rest[len(key) + 1:]


def group_vec(exam: dict, group: str) -> np.ndarray:
    return np.concatenate([vec(exam[c]) for c in GROUPS[group] if c in exam]) if any(c in exam for c in GROUPS[group]) \
        else np.array([], dtype=np.int8)


def seed_cell(values: list[float], boots: np.ndarray | None = None, d: np.ndarray | None = None) -> dict:
    v = np.array(values, dtype=float)
    out = {"values": [round(x, 1) for x in values], "mean": round(float(v.mean()), 1)}
    if d is not None:  # per-item difference of the seed mean: the paired test over items
        out["p"] = signflip(d) if np.any(d != 0) else 1.0
        out["gained"], out["lost"] = int((d > 0).sum()), int((d < 0).sum())
    if len(v) > 1:
        sd = float(v.std(ddof=1))
        h = float(tdist.ppf(0.975, len(v) - 1)) * sd / np.sqrt(len(v))
        out.update({"sd": round(sd, 1), "ci_seeds": [round(float(v.mean()) - h, 1), round(float(v.mean()) + h, 1)]})
    if boots is not None:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["ci_questions"] = [round(float(lo), 1), round(float(hi), 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-question", default=str(Path(__file__).resolve().parents[3] / "data" / "per_question"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "data" / "stats.json"))
    a = ap.parse_args()
    pq = Path(a.per_question)
    bfcl = json.loads((pq / "bfcl.json").read_text())["exams"]
    probe = json.loads((pq / "probe.json").read_text())["runs"]
    practice = json.loads((pq / "practice.json").read_text())["runs"]

    # ---- BFCL against the floor ----
    base_rows, group_rows = [], []
    for name, ex in sorted(bfcl.items()):
        if name.startswith("floor-"):
            continue
        exp, key, run = exam_key(name)
        floor = bfcl[f"floor-{key}"]
        r = {"exam": name, "exp": exp, "method": METHOD[exp], "model": key, "run": run,
             **paired(vec(floor["multi_turn_base"]), vec(ex["multi_turn_base"]))}
        base_rows.append(r)
        for g in GROUPS:
            fa, fb = group_vec(floor, g), group_vec(ex, g)
            if len(fa) and len(fb) == len(fa):
                group_rows.append({"exam": name, "exp": exp, "method": METHOD[exp], "model": key, "run": run,
                                   "group": g, **paired(fa, fb)})
    holm(base_rows)
    holm(group_rows)

    # ---- probe: final state against nothing learnt ----
    probe_rows = []
    for k, s in sorted(probe.items()):
        parts = k.split("/")
        if parts[0] == "base" or parts[-1] != "1200":
            continue
        exp, key, run = parts[0], parts[1], parts[2]
        probe_rows.append({"run": f"{exp}/{key}/{run}", "exp": exp, "method": METHOD[exp], "model": key, "arm": run,
                           **paired(vec(probe[f"base/{key}"]), vec(s))})
    holm(probe_rows)

    # ---- practice: learning run against E15's frozen model, task by task ----
    prac_rows = []
    for k, s in sorted(practice.items()):
        exp, key, run = k.split("/")
        if run.startswith("frozen"):
            continue
        prac_rows.append({"run": k, "exp": exp, "method": METHOD[exp], "model": key, "arm": run,
                          **paired(vec(practice[f"e15/{key}/frozen-s15"]), vec(s))})
    holm(prac_rows)

    # ---- methods against each other (seed 15, same model and arm) ----
    method_rows = []
    for key in MODELS:
        for arm in ("teach-s15", "judge-s15"):
            for hi_exp, lo_exp in (("e16", "e17"), ("e17", "e15")):
                pa, pb = probe.get(f"{lo_exp}/{key}/{arm}/1200"), probe.get(f"{hi_exp}/{key}/{arm}/1200")
                if pa and pb:
                    method_rows.append({"compare": f"{METHOD[hi_exp]} vs {METHOD[lo_exp]}", "on": "probe", "model": key,
                                        "arm": arm, **paired(vec(pa), vec(pb))})
                ea, eb = bfcl.get(f"{lo_exp}-{key}-{arm}"), bfcl.get(f"{hi_exp}-{key}-{arm}")
                if ea and eb:
                    method_rows.append({"compare": f"{METHOD[hi_exp]} vs {METHOD[lo_exp]}", "on": "bfcl_base",
                                        "model": key, "arm": arm,
                                        **paired(vec(ea["multi_turn_base"]), vec(eb["multi_turn_base"]))})
    holm(method_rows)

    # ---- E19's three-seed cells ----
    cells = []
    for key, arm in [("qwen3-1.7b", "teach"), ("qwen3-1.7b", "teach-think"), ("qwen3-1.7b", "judge"),
                     ("lfm2.5-1.2b", "teach"), ("qwen3.5-0.8b", "teach"), ("llama3.2-1b", "teach"),
                     ("gemma3-1b", "teach")]:
        exams = [f"{'e16' if s == 15 and arm != 'teach-think' else 'e19'}-{key}-{arm}-s{s}" for s in (15, 16, 17)]
        runs = [f"{'e16' if s == 15 and arm != 'teach-think' else 'e19'}/{key}/{arm}-s{s}" for s in (15, 16, 17)]
        if not all(e in bfcl for e in exams):
            continue
        floor = vec(bfcl[f"floor-{key}"]["multi_turn_base"]).astype(float)
        X = np.stack([vec(bfcl[e]["multi_turn_base"]).astype(float) for e in exams])
        n = X.shape[1]
        idx = RNG.integers(0, n, size=(B, n))
        dmean = (X.mean(axis=0) - floor)
        base_boot = dmean[idx].mean(axis=1) * 100
        P0 = vec(probe[f"base/{key}"]).astype(float)
        PX = np.stack([vec(probe[f"{r}/1200"]).astype(float) for r in runs])
        pboot = (PX.mean(axis=0) - P0)[RNG.integers(0, 200, size=(B, 200))].mean(axis=1) * 100
        F = vec(practice[f"e15/{key}/frozen-s15"]).astype(float)
        GX = np.stack([vec(practice[r]).astype(float) for r in runs])
        gboot = (GX.mean(axis=0) - F)[RNG.integers(0, 1200, size=(B, 1200))].mean(axis=1) * 100
        cells.append({"model": key, "arm": arm, "exams": exams,
                      "bfcl_base_vs_floor": seed_cell([100 * (x.mean() - floor.mean()) for x in X], base_boot, dmean),
                      "probe_vs_start": seed_cell([100 * (x.mean() - P0.mean()) for x in PX], pboot, PX.mean(axis=0) - P0),
                      "practice_vs_frozen": seed_cell([100 * (x.mean() - F.mean()) for x in GX], gboot,
                                                      GX.mean(axis=0) - F)})
    holm([c[m] for c in cells for m in ("bfcl_base_vs_floor", "probe_vs_start", "practice_vs_frozen")])

    # ---- LoRA arms against each other (computed last, so the families above draw the same bootstrap samples) ----
    def exam_name(key, arm, s):
        return f"{'e16' if s == 15 and arm != 'teach-think' else 'e19'}-{key}-{arm}-s{s}"

    def run_name(key, arm, s):
        return f"{'e16' if s == 15 and arm != 'teach-think' else 'e19'}/{key}/{arm}-s{s}"

    def paired_seeds(As, Bs) -> dict:
        m = np.all([a >= 0 for a in As] + [b >= 0 for b in Bs], axis=0)
        D = np.stack([b[m].astype(float) - a[m].astype(float) for a, b in zip(As, Bs)])
        d = D.mean(axis=0)
        n = len(d)
        boots = d[RNG.integers(0, n, size=(B, n))].mean(axis=1) * 100
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"n": n, "values": [round(100 * float(x.mean()), 1) for x in D], "diff": round(100 * float(d.mean()), 1),
                "ci": [round(float(lo), 1), round(float(hi), 1)], "gained": int((d > 0).sum()),
                "lost": int((d < 0).sum()), "p": signflip(d) if np.any(d != 0) else 1.0}

    arm_rows = []
    for key, hi_arm, lo_arm, seeds in [("lfm2.5-1.2b", "judge", "teach", (15,)), ("qwen3.5-0.8b", "judge", "teach", (15,)),
                                       ("qwen3-1.7b", "judge", "teach", (15, 16, 17)),
                                       ("qwen3-1.7b", "teach-think", "teach", (15, 16, 17))]:
        lo_ex = [bfcl[exam_name(key, lo_arm, s)] for s in seeds]
        hi_ex = [bfcl[exam_name(key, hi_arm, s)] for s in seeds]
        measures = [("bfcl_base", [vec(e["multi_turn_base"]) for e in lo_ex], [vec(e["multi_turn_base"]) for e in hi_ex])]
        measures += [(f"bfcl_{g}", [group_vec(e, g) for e in lo_ex], [group_vec(e, g) for e in hi_ex]) for g in GROUPS]
        measures.append(("probe", [vec(probe[f"{run_name(key, lo_arm, s)}/1200"]) for s in seeds],
                         [vec(probe[f"{run_name(key, hi_arm, s)}/1200"]) for s in seeds]))
        for on, As, Bs in measures:
            stats = paired(As[0], Bs[0]) if len(seeds) == 1 else paired_seeds(As, Bs)
            arm_rows.append({"compare": f"{hi_arm} vs {lo_arm}", "on": on, "model": key, "seeds": list(seeds), **stats})
    holm(arm_rows)

    out = {"about": __doc__.split("\n\n")[0], "method": {"test": "McNemar exact (two-sided)", "interval":
           "95% paired bootstrap over questions, 10,000 resamples, seed 20", "correction": "Holm within each family",
           "seed_cells": "mean over 3 seeds; ci_seeds = t interval over seeds; ci_questions = bootstrap over questions"},
           "bfcl_base": base_rows, "bfcl_groups": group_rows, "probe": probe_rows, "practice": prac_rows,
           "methods": method_rows, "seed_cells": cells, "arms": arm_rows}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for fam in ("bfcl_base", "bfcl_groups", "probe", "practice", "methods", "seed_cells", "arms"):
        rows = out[fam] if fam != "seed_cells" else [c[m] for c in cells for m in
                                                      ("bfcl_base_vs_floor", "probe_vs_start", "practice_vs_frozen")]
        print(f"{fam}: {len(rows)} tests, {sum(r['significant'] for r in rows)} significant after Holm")


if __name__ == "__main__":
    main()

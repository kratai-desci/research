"""Every statistical test in the paper (Appendix B), recomputed from data/ alone.

  uv run --with scipy python src/stats.py [data dir]     # writes data/stats.json and prints one row per test

The unit is a life (one seed): each life gives one number, so the seeds are the samples. With 3-5 lives per group,
every two-group comparison is reported both as a Welch t-test and a Mann-Whitney U test (the data are not
bell-shaped: sense lives either stay near the oracle or collapse to 0). The primary claims are Holm-corrected as one
family.
"""

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
FAILED = 50  # a WebShop attempt below reward 0.5 counts as failed (the oracle's own line)

LIVES = list(csv.DictReader(open(DATA / "lives.csv")))
EXAMS = list(csv.DictReader(open(DATA / "exams.csv")))
ATTEMPTS = list(csv.DictReader(gzip.open(DATA / "attempts.csv.gz", "rt", encoding="utf-8")))


def lives(lab_experiment: str) -> list[dict]:
    return [l for l in LIVES if l["lab_experiment"] == lab_experiment]


def exam(names: set[str], day: int, splits=("probe",)) -> dict[str, float]:
    """Each life's mean exam score on one day, over all tasks of the given exam splits."""
    acc = defaultdict(lambda: [0.0, 0])
    for e in EXAMS:
        if e["life"] in names and int(e["day"]) == day and e["split"] in splits:
            acc[e["life"]][0] += float(e["mean_score"]) * int(e["tasks"])
            acc[e["life"]][1] += int(e["tasks"])
    return {k: s / n for k, (s, n) in acc.items()}


def lessons(names: set[str], last_day: int) -> dict[str, tuple[int, int, int]]:
    """Per life: lessons up to last_day, how many were failed attempts, and how many failed lessons on night 1."""
    out = defaultdict(lambda: [0, 0, 0])
    for a in ATTEMPTS:
        if a["life"] in names and a["lesson"] == "yes" and int(a["day"]) <= last_day:
            bad = a["score"] != "" and float(a["score"]) < FAILED
            o = out[a["life"]]
            o[0] += 1; o[1] += bad; o[2] += bad and int(a["day"]) == 1
    return {k: tuple(v) for k, v in out.items()}


def group(rs, judge, day, splits=("probe",)):
    return list(exam({l["life"] for l in rs if l["judge"] == judge}, day, splits).values())


results = []


def compare(claim, a_name, a, b_name, b, primary=False):
    w = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    m = lambda xs: f"{sum(xs) / len(xs):.1f} ± {stats.tstd(xs):.1f} (n={len(xs)})"
    results.append({"claim": claim, "test": "Welch t / Mann-Whitney U", "groups": f"{a_name} {m(a)} vs {b_name} {m(b)}",
                    "statistic": f"t = {w.statistic:.2f}", "p": float(w.pvalue), "p_mw": float(u.pvalue), "primary": primary})


# Experiment 1: five judges, day 14
e1 = lives("growing-from-use-14d")
compare("Exp 1: oracle above self on day 14", "oracle", group(e1, "oracle", 14), "self", group(e1, "self", 14), primary=True)
compare("Exp 1: oracle above both on day 14", "oracle", group(e1, "oracle", 14), "both", group(e1, "both", 14), primary=True)
compare("Exp 1: self against no learning on day 14", "self", group(e1, "self", 14), "none", group(e1, "none", 14))

# Experiment 2: sense v1, day 14
e2 = lives("feedback-sense-14d")
compare("Exp 2: oracle against sense v1 on day 14", "oracle", group(e2, "oracle", 14), "sense", group(e2, "sense", 14))
compare("Exp 2: sense v1 against both on day 14", "sense", group(e2, "sense", 14), "both", group(e2, "both", 14))

# Experiment 3: realistic users
e3 = lives("realistic-user-14d")
compare("Exp 3: oracle above sense v2 on day 14", "oracle", group(e3, "oracle", 14), "sense", group(e3, "sense", 14))

# Experiment 4: sense v3, day 14 and day 60
e4 = lives("clear-verdict-14d")
compare("Exp 4: oracle against sense v3 on day 14", "oracle", group(e4, "oracle", 14), "sense", group(e4, "sense", 14))
compare("Exp 4: oracle above sense v3 on day 60", "oracle", group(e4, "oracle", 60), "sense", group(e4, "sense", 60), primary=True)

# 5.6: share of failed lessons against the day-14 exam, every 14-day WebShop life that learned at least once
pool = []
for exp in ("growing-from-use-14d", "feedback-sense-14d", "realistic-user-14d", "lesson-once-14d", "clear-verdict-14d", "tree-rings-14d"):
    names = {l["life"] for l in lives(exp)}
    ex, ls = exam(names, 14), lessons(names, 14)
    for l in lives(exp):
        n, bad, _ = ls.get(l["life"], (0, 0, 0))
        if l["life"] in ex and n > 0:
            pool.append({"judge": l["judge"], "share": bad / n, "score": ex[l["life"]]})
for label, sel in (("all lives", pool), ("lives not judged by the oracle", [p for p in pool if p["judge"] != "oracle"])):
    rho = stats.spearmanr([p["share"] for p in sel], [p["score"] for p in sel])
    results.append({"claim": f"5.6: share of failed lessons against day-14 exam, {label}", "test": "Spearman rank correlation",
                    "groups": f"n={len(sel)} lives", "statistic": f"rho = {rho.statistic:.2f}", "p": float(rho.pvalue),
                    "primary": label == "lives not judged by the oracle"})

# 5.6: a failed lesson on night 1 against collapse (day-14 exam of 3 or less), sense lives of Experiments 2-4
sense = []
for exp in ("feedback-sense-14d", "realistic-user-14d", "lesson-once-14d", "clear-verdict-14d"):
    rs = [l for l in lives(exp) if l["judge"] == "sense"]
    names = {l["life"] for l in rs}
    ex, ls = exam(names, 14), lessons(names, 1)
    sense += [{"bad1": ls.get(l["life"], (0, 0, 0))[1] > 0, "collapsed": ex[l["life"]] <= 3} for l in rs if l["life"] in ex]
t = [[sum(s["bad1"] and s["collapsed"] for s in sense), sum(s["bad1"] and not s["collapsed"] for s in sense)],
     [sum(not s["bad1"] and s["collapsed"] for s in sense), sum(not s["bad1"] and not s["collapsed"] for s in sense)]]
f = stats.fisher_exact(t)
results.append({"claim": "5.6: a failed lesson on night 1 predicts collapse (sense lives)", "test": "Fisher exact",
                "groups": f"collapsed {t[0][0]} of {t[0][0] + t[0][1]} with one, {t[1][0]} of {t[1][0] + t[1][1]} without",
                "statistic": f"odds ratio = {f.statistic}", "p": float(f.pvalue), "primary": True})

# 5.4: sense v3 against v2 reading give-ups and polite complaints as a yes (the same recorded reactions)
s = json.loads((DATA / "sense" / "sense3_train-status.json").read_text())["eval3"]
yes = lambda key: sum(int(s[key]["yes_by_behaviour"][b].split("/")[0]) for b in ("give_up", "polite_unhappy"))
n = sum(int(s["live"]["yes_by_behaviour"][b].split("/")[1]) for b in ("give_up", "polite_unhappy"))
f = stats.fisher_exact([[yes("live_v2"), n - yes("live_v2")], [yes("live"), n - yes("live")]])
results.append({"claim": "5.4: v3 misreads give-ups and polite complaints less than v2", "test": "Fisher exact (unpaired; conservative)",
                "groups": f"v2 {yes('live_v2')} of {n}, v3 {yes('live')} of {n}", "statistic": f"odds ratio = {f.statistic:.1f}",
                "p": float(f.pvalue), "primary": True})

# Experiment 5: 1.7B, oracle against sense v3, day-30 exam over all three domains
e5 = lives("tau2-learning-30d")
ALL = ("probe", "probe-transfer")
compare("Exp 5: 1.7B oracle against 1.7B sense v3 on day 30 (all exam tasks)", "oracle", group(e5, "oracle-1.7B", 30, ALL),
        "sense", group(e5, "sense-1.7B", 30, ALL))
names = {l["life"] for l in e5 if l["judge"] == "sense-1.7B"}
d0, d30 = exam(names, 0, ALL), exam(names, 30, ALL)
both = sorted(set(d0) & set(d30))
p = stats.ttest_rel([d0[k] for k in both], [d30[k] for k in both])
results.append({"claim": "Exp 5: 1.7B sense v3 lives fell from day 0 to day 30 (all exam tasks)", "test": "paired t",
                "groups": f"day 0 {sum(d0[k] for k in both) / len(both):.1f} vs day 30 {sum(d30[k] for k in both) / len(both):.1f} (n={len(both)} lives)",
                "statistic": f"t = {p.statistic:.2f}", "p": float(p.pvalue), "primary": False})

# Holm correction over the primary claims
prim = sorted([r for r in results if r["primary"]], key=lambda r: r["p"])
running = 0.0
for i, r in enumerate(prim):
    running = max(running, min(1.0, (len(prim) - i) * r["p"]))
    r["p_holm"] = running

fmt = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
for r in results:
    extra = f"; Mann-Whitney p {fmt(r['p_mw'])}" if "p_mw" in r else ""
    holm = f"; Holm p {fmt(r['p_holm'])}" if "p_holm" in r else ""
    print(f"{'*' if r['primary'] else ' '} {r['claim']}\n    {r['test']}: {r['groups']}; {r['statistic']}, p {fmt(r['p'])}{extra}{holm}")
(DATA / "stats.json").write_text(json.dumps(results, indent=1))

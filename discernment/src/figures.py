"""Draw the paper's data figures from data/ into paper/figures/ (PDF for LaTeX, PNG for a quick look).

  uv run --with matplotlib python src/figures.py [data dir] [out dir]

Fig 1 and the sense diagram are diagrams (src/diagrams.py); the others are drawn here from the rows in data/.
"""

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, DARK, GREY, LIGHT, PINK = "#2a6fd6", "#1a9e6e", "#444444", "#9a9a90", "#c9c9c0", "#d0507a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.bbox": "tight", "font.family": "sans-serif"})

LIVES = list(csv.DictReader(open(DATA / "lives.csv")))
EXAMS = list(csv.DictReader(open(DATA / "exams.csv")))
ATTEMPTS = list(csv.DictReader(gzip.open(DATA / "attempts.csv.gz", "rt", encoding="utf-8")))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(OUT / f"{name}.pdf")


def curves(exp, splits=("probe",), domain=None):
    """judge -> {day: [score of each life]} (mean over all tasks of the splits, per life and day)."""
    judge = {l["life"]: l["judge"] for l in LIVES if l["lab_experiment"] == exp}
    acc = defaultdict(lambda: [0.0, 0])
    for e in EXAMS:
        if e["life"] in judge and e["split"] in splits and (domain is None or e["domain"] == domain):
            k = (e["life"], int(e["day"]))
            acc[k][0] += float(e["mean_score"]) * int(e["tasks"]); acc[k][1] += int(e["tasks"])
    out = defaultdict(lambda: defaultdict(list))
    for (life, day), (s, n) in acc.items():
        out[judge[life]][day].append(s / n)
    return out


def band_plot(ax, series, styles, label_names=None):
    ends = []
    for j, (color, lw, band, dash) in styles.items():
        if j not in series:
            continue
        days = sorted(series[j])
        mean = [sum(series[j][d]) / len(series[j][d]) for d in days]
        if band:
            ax.fill_between(days, [min(series[j][d]) for d in days], [max(series[j][d]) for d in days], color=color, alpha=0.13, lw=0)
        ax.plot(days, mean, color=color, lw=lw, ls=dash, marker="o", ms=2.2)
        ends.append([mean[-1], days[-1], f"{(label_names or {}).get(j, j)} {mean[-1]:.0f}", color])
    ends.sort()
    for k in range(1, len(ends)):  # keep end labels at least 5 points apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 5)
    for y, x, text, color in ends:
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", color=color, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("WebShop exam (0-100)")
    ax.set_xlabel("day")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)


# Fig 2: Experiment 1, (a) five judges over 14 days and (b) a year of life, side by side
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.5})
band_plot(a1, curves("growing-from-use-14d"), {"oracle": (BLUE, 2.2, True, "-"), "self": (DARK, 2.0, True, "-"), "both": (GREY, 2.0, False, "-"),
                  "stated verdict": (LIGHT, 1.4, False, "-"), "none": (GREY, 1.2, False, "--")}, {"stated verdict": "stated"})
a1.set_xlim(0, 14); a1.set_title("(a) Experiment 1-A: 14 days", fontsize=9)
band_plot(a2, curves("lifespan-365d"), {"oracle": (BLUE, 1.8, False, "-"), "both": (DARK, 1.8, True, "-")})
a2.set_xlim(0, 365); a2.set_ylabel(""); a2.set_title("(b) Experiment 1-B: a year", fontsize=9)
save(fig, "fig2_exp1_judges")

# Fig 10: Experiments 2 and 3, sense v1 and sense v2 over 14 days
V1, V2 = "#8a6fbf", "#c8553d"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True)
s = curves("feedback-sense-14d")
band_plot(a1, s, {"oracle": (BLUE, 2.2, True, "-"), "sense": (V1, 2.2, True, "-"), "both": (GREY, 1.8, False, "-")}, {"sense": "sense v1"})
a1.set_xlim(0, 14); a1.set_xticks([0, 7, 14])
a1.set_title("(a) Experiment 2: sense v1 kept rising", fontsize=9)
s = curves("realistic-user-14d")
band_plot(a2, s, {"oracle": (BLUE, 2.2, True, "-"), "both": (GREY, 1.8, False, "-")})
lives = defaultdict(dict)
for e in EXAMS:
    if e["life"].startswith("realistic-user-14d-sense-") and e["split"] == "probe":
        lives[e["life"]][int(e["day"])] = float(e["mean_score"])
for k, (life, by_day) in enumerate(sorted(lives.items())):
    days = sorted(by_day)
    a2.plot(days, [by_day[d] for d in days], color=V2, lw=1.1, marker="o", ms=1.6, label="sense v2, each life" if k == 0 else None)
a2.legend(loc="upper left", fontsize=7.5, frameon=False)
a2.set_xlim(0, 14); a2.set_xticks([0, 7, 14]); a2.set_ylabel("")
a2.set_title("(b) Experiment 3: 3 of 5 sense v2 lives collapsed", fontsize=9)
fig.tight_layout(w_pad=3)
save(fig, "fig10_exp2_exp3")

# Fig 4: Experiment 4, sense v3 over 60 days
s = curves("clear-verdict-14d")
fig, ax = plt.subplots(figsize=(5.6, 2.45))
band_plot(ax, s, {"oracle": (BLUE, 2.2, True, "-"), "sense": (GREEN, 2.2, True, "-"), "both": (GREY, 1.8, False, "-")},
          {"sense": "sense v3"})
ax.set_xlim(0, 60)
save(fig, "fig4_sense_v3_60_days")

# Fig 5: Experiment 5, tau2-bench exams per domain
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)
for ax, dom in zip(axes, ("airline", "retail", "telecom")):
    split = ("probe-transfer",) if dom == "telecom" else ("probe",)
    s = curves("tau2-learning-30d", split, dom)
    for j, color, alpha in (("oracle-1.7B", BLUE, 1), ("sense-1.7B", GREEN, 1), ("oracle-0.6B", BLUE, 0.4), ("sense-0.6B", GREEN, 0.4)):
        days = sorted(s[j])
        ax.plot(days, [sum(s[j][d]) / len(s[j][d]) for d in days], color=color, alpha=alpha, lw=1.8, marker="o", ms=2,
                label=j.replace("sense", "sense v3"))
    ax.set_title(f"{dom} ({'never learned' if dom == 'telecom' else 'learned'})", fontsize=9)
    ax.set_xlim(0, 30); ax.set_ylim(0, 50); ax.set_xlabel("day"); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
axes[0].set_ylabel("exam tasks fully done (%)")
axes[2].legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "fig5_tau2_domains")

# Fig 6: share of failed lessons against the day-14 exam, 97 lives
EXPS = {"growing-from-use-14d": "Exp 1-A", "tree-rings-14d": "Exp 1-C", "feedback-sense-14d": "Exp 2",
        "realistic-user-14d": "Exp 3", "lesson-once-14d": "Exp 3b", "clear-verdict-14d": "Exp 4"}
life = {l["life"]: l for l in LIVES if l["lab_experiment"] in EXPS}
les = defaultdict(lambda: [0, 0])
for a in ATTEMPTS:
    if a["life"] in life and a["lesson"] == "yes" and int(a["day"]) <= 14:
        les[a["life"]][0] += 1; les[a["life"]][1] += a["score"] != "" and float(a["score"]) < 50
day14 = {e["life"]: float(e["mean_score"]) for e in EXAMS if e["life"] in life and e["day"] == "14" and e["split"] == "probe"}


def kind(l):
    j = l["judge"]
    if j == "sense":
        return {"clear-verdict-14d": "sense v3", "feedback-sense-14d": "sense v1"}.get(l["lab_experiment"], "sense v2")
    return j if j in ("oracle", "stated verdict") else "self-judging"


fig, ax = plt.subplots(figsize=(4.3, 2.5))
colors = {"self-judging": LIGHT, "stated verdict": PINK, "sense v1": "#8a6fbf", "sense v2": "#c8553d", "sense v3": GREEN, "oracle": BLUE}
for k, c in colors.items():
    pts = [(100 * les[n][1] / les[n][0], day14[n]) for n, l in life.items() if kind(l) == k and les[n][0] > 0 and n in day14]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18, color=c, alpha=0.8, label=k.replace("stated verdict", "stated"), lw=0)
ax.set_xlabel("share of lessons that were failed attempts (%)"); ax.set_ylabel("exam on day 14")
ax.set_xlim(-3, 103); ax.set_ylim(-3, 70); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
save(fig, "fig6_failed_lessons")

# Fig 3: sense v2 against v3 on 2,622 recorded reactions
ev = json.loads((DATA / "sense" / "sense3_train-status.json").read_text())["eval3"]
order = [("praise", "praise"), ("accept", "accept"), ("settle", "settle for less"), ("adjust", "ask for a change"),
         ("polite_unhappy", "polite complaint"), ("give_up", "give up")]
# reject, repeat the request, move on without words and silence were read as a yes 0 times by both versions (see the caption)
for b in ("reject", "repeat", "move_on", "silent"):
    assert ev["live"]["yes_by_behaviour"][b].startswith("0/") and ev["live_v2"]["yes_by_behaviour"][b].startswith("0/"), b
pct = lambda key, b: 100 * int(ev[key]["yes_by_behaviour"][b].split("/")[0]) / int(ev[key]["yes_by_behaviour"][b].split("/")[1])
fig, ax = plt.subplots(figsize=(5.0, 1.95))
ys = range(len(order))[::-1]
ax.barh([y + 0.2 for y in ys], [pct("live_v2", b) for b, _ in order], height=0.38, color=LIGHT, label="sense v2")
ax.barh([y - 0.2 for y in ys], [pct("live", b) for b, _ in order], height=0.38, color=GREEN, label="sense v3 (waits for a clear verdict)")
ax.set_yticks(list(ys), [n for _, n in order])
for t, (b, _) in zip(ax.get_yticklabels(), order):
    t.set_fontweight("bold" if b in ("polite_unhappy", "give_up") else "normal")
ax.set_xlim(0, 105); ax.set_xlabel("share of each reaction read as a yes (%)")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
save(fig, "fig3_sense_v2_v3")

gaps = json.loads((DATA / "sense" / "sense3_status.json").read_text())["check"]["gaps"]
kinds = [("accepts", "plain acceptance", "yes"), ("builds_on", "builds on the result", "yes"), ("strong_reject", "firm rejection", "no"),
         ("not_my_style", "'not my style'", "no"), ("polite_unhappy", "politely unhappy", "no"), ("vague", "vague", "not clear"),
         ("settling", "settling", "not clear"), ("sarcastic", "sarcastic", "not clear"), ("thanks_for_effort", "thanks for trying (v3)", "not clear")]
loto = {}
for p in sorted((DATA / "sense").glob("sense2_train-no-*-status.json")):
    st = json.loads(p.read_text())
    loto[st["hold_out"]] = 100 * st["eval2"]["held_out_task"]["false_accept"]
# Fig 8a: the v3 teacher's reading of the reactions it wrote (shown with sense v3, 3.2)
fig, a1 = plt.subplots(figsize=(5.2, 3.4))
ys = list(range(len(kinds)))[::-1]
for y, (k, name, _) in zip(ys, kinds):
    c = gaps[k]; n = sum(c.values())
    yes = 100 * (c.get("80", 0) + c.get("100", 0)) / n; no = 100 * (c.get("0", 0) + c.get("20", 0)) / n
    a1.barh(y, yes, color=BLUE); a1.barh(y, no, left=yes, color=GREY); a1.barh(y, 100 - yes - no, left=yes + no, color="#e3e3dc")
    a1.text(102, y, f"{yes:.0f}% yes", va="center", fontsize=7.5, color=BLUE if yes >= 10 and k not in ("accepts", "builds_on") else DARK)
a1.set_yticks(ys, [f"{n}  [{w}]" for _, n, w in kinds], fontsize=7.5)
a1.set_xlim(0, 125); a1.set_xticks([0, 50, 100]); a1.set_xlabel("teacher's reading (%): yes / no / not clear")
from matplotlib.patches import Patch
a1.legend(handles=[Patch(color=BLUE, label="yes"), Patch(color=GREY, label="no"), Patch(color="#e3e3dc", label="not clear")], fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.4, -0.34))
fig.tight_layout()
save(fig, "fig8a_teacher_written")

# Fig 8b: sense v2 on task types it never saw (shown with sense v2, 3.2)
fig, a2 = plt.subplots(figsize=(5.2, 3.0))
items = sorted(loto.items(), key=lambda kv: kv[1])
a2.barh(range(len(items)), [v for _, v in items], color=GREY)
a2.set_yticks(range(len(items)), [k for k, _ in items], fontsize=7.5)
a2.axvline(3, color=DARK, ls=":", lw=1); a2.text(3, len(items) - 0.4, "3% target", fontsize=7, ha="center")
for i, (_, v) in enumerate(items):
    a2.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=7.5)
a2.set_xlim(0, 10); a2.set_xlabel("unseen type's noes read as a yes (%)")
fig.tight_layout()
save(fig, "fig8b_v2_unseen_tasks")

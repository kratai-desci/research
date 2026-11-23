"""Figures 2-4 of Paper 2, drawn from E20's per-question results and tests (data/, from src/lab/e20/).

    uv run --with matplotlib python paper/learning-potential/figures/make_figures.py

Writes probe_curves.pdf, reasoning_curves.pdf and side_effects.pdf next to this file. Colours: the validated reference
palette (categorical blue, orange, aqua; diverging blue-red with a grey midpoint).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parent
PROBE = json.loads((DATA / "per_question/probe.json").read_text())["runs"]
STATS = json.loads((DATA / "stats.json").read_text())

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"
INK, MUTED = "#0b0b0b", "#52514e"
MODELS = [("qwen3-1.7b", "Qwen3-1.7B"), ("lfm2.5-1.2b", "LFM2.5-1.2B"), ("qwen3.5-0.8b", "Qwen3.5-0.8B"),
          ("llama3.2-1b", "Llama-3.2-1B"), ("falcon3-1b", "Falcon3-1B"), ("gemma3-1b", "Gemma-3-1B")]
STEPS = list(range(100, 1201, 100))

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})


def score(bits: str) -> float:
    return 100 * bits.count("1") / len(bits)


def curve(run: str, model: str):
    """Probe score at 0 (the model with nothing learnt) and every 100 tasks; None if the run has no snapshots."""
    pts = [score(PROBE[f"base/{model}"])]
    for n in STEPS:
        key = f"{run}/{n}"
        if key not in PROBE:
            return None
        pts.append(score(PROBE[key]))
    return pts


def seeds(exp_run_pairs, model):
    return [c for c in (curve(f"{e}/{model}/{r}", model) for e, r in exp_run_pairs) if c]


X = [0] + STEPS


def band(ax, curves, color, label, dashed=False):
    if not curves:
        return
    mean = [sum(v) / len(v) for v in zip(*curves)]
    ls = (0, (4, 2)) if dashed else "-"
    if len(curves) > 1:
        lo = [min(v) for v in zip(*curves)]
        hi = [max(v) for v in zip(*curves)]
        ax.fill_between(X, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(X, mean, color=color, linewidth=1.4 if not dashed else 1.0, linestyle=ls, label=label)


# ---- Per-method probe figures: notes, retrieval and LoRA drawn identically ----
# Six panels (one per model), probe solved (%) on the same 0-80 axis, oracle teaching solid and oracle judging dashed
# in the method's colour, the frozen model dotted grey. Cells with three seeds show the seed mean and range.
TEACH = {"e15": [("e15", "teach-s15")], "e17": [("e17", "teach-s15")],
         "e16": [("e16", "teach-s15"), ("e19", "teach-s16"), ("e19", "teach-s17")]}
JUDGE = {"e15": [("e15", "judge-s15")], "e17": [("e17", "judge-s15")],
         "e16": [("e16", "judge-s15"), ("e19", "judge-s16"), ("e19", "judge-s17")]}


def method_figure(exp, color, fname):
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.9), sharey=True)
    for ax, (key, label) in zip(axes, MODELS):
        ax.axhline(score(PROBE[f"base/{key}"]), color=GREY, linewidth=0.8, linestyle=(0, (1, 1.5)), label="frozen")
        band(ax, seeds(TEACH[exp], key), color, "oracle teaching")
        band(ax, seeds(JUDGE[exp], key), color, "oracle judging", dashed=True)
        ax.set_title(label, fontsize=7.5, color=INK, loc="left", pad=2)
        ax.set_ylim(0, 80)
        ax.set_xlim(0, 1200)
        ax.set_xticks([0, 600, 1200])
        ax.set_xlabel("practice tasks", fontsize=7)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
    axes[0].set_ylabel("probe solved (%)", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.08), handlelength=2.2)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


method_figure("e15", AQUA, "notes_curves.pdf")
method_figure("e17", ORANGE, "retrieval_curves.pdf")
method_figure("e16", BLUE, "lora_curves.pdf")

# ---- Transfer: gain in practice, on the probe and on BFCL multi-turn base, oracle teaching ----
practice = {r["run"]: r for r in STATS["practice"]}
probe_t = {r["run"]: r for r in STATS["probe"]}
bfcl_t = {r["exam"]: r for r in STATS["bfcl_base"]}
cells3 = {(c["model"], c["arm"]): c for c in STATS["seed_cells"]}


def stage_values(exp, key):
    """(practice, probe, BFCL) as (change, significant); LoRA teaching uses the three-seed cell where there is one."""
    if exp == "e16" and (key, "teach") in cells3:
        c = cells3[(key, "teach")]
        return [(c[k]["mean"], c[k]["significant"]) for k in ("practice_vs_frozen", "probe_vs_start",
                                                               "bfcl_base_vs_floor")]
    run = f"{exp}/{key}/teach-s15"
    out = []
    for table, k in ((practice, run), (probe_t, run), (bfcl_t, f"{exp}-{key}-teach-s15")):
        r = table.get(k)
        out.append((r["diff"], r["significant"]) if r else (float("nan"), False))
    return out


# Horizontal grouped bars: one group per run (oracle teaching) that moved, three bars for practice, probe and BFCL
# multi-turn base in three ordered shades of one hue; hatched = not significant. Notes, retrieval for the zero-floor
# models and retrieval for Qwen3-1.7B (all within a few points of zero) are named in the caption instead.
ROWS_T = [("e16", k) for k, _ in MODELS] + [("e17", k) for k in ("qwen3.5-0.8b", "lfm2.5-1.2b")]
rows = [(exp, key, stage_values(exp, key)) for exp, key in ROWS_T]
rows.sort(key=lambda r: (r[0] != "e16", -r[2][1][0]))  # LoRA first, then by probe gain
SHADES = [("practice", "#86b6ef"), ("probe", "#3987e5"), ("BFCL", "#184f95")]  # validated ordinal, light to dark
fig, ax = plt.subplots(figsize=(3.5, 3.4))
h = 0.26
ticks, labels = [], []
for i, (exp, key, vals) in enumerate(rows):
    y0 = len(rows) - 1 - i
    for j, ((v, sg), (name, color)) in enumerate(zip(vals, SHADES)):
        y = y0 + h - j * h
        ax.barh(y, v, height=h * 0.92, color=color if sg else "white", edgecolor=color, linewidth=0.8,
                hatch=None if sg else "//////", label=name if i == 0 else None)
        ax.text(v + (0.8 if v >= 0 else -0.8), y, f"{v:+.0f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=5.0, color=INK)
    ticks.append(y0)
    labels.append(f"{'LoRA' if exp == 'e16' else 'retrieval'} · {dict(MODELS)[key]}")
ax.axvline(0, color=MUTED, linewidth=0.7)
ax.set_yticks(ticks, labels, fontsize=6.3)
ax.set_xlim(-12, 64)
ax.set_xlabel("gain (points)", fontsize=7)
ax.grid(axis="x", color="#e6e5e1", linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=1.2)
fig.tight_layout()
fig.savefig(OUT / "transfer.pdf", bbox_inches="tight")
plt.close(fig)

# ---- Reasoning model: Qwen3-1.7B's three LoRA arms, in the method figures' style (one panel per arm, one line per seed)
fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.9), sharey=True)
key = "qwen3-1.7b"
arms = [("teaching", [("e16", "teach-s15"), ("e19", "teach-s16"), ("e19", "teach-s17")], False),
        ("reasoning kept", [("e19", "teach-think-s15"), ("e19", "teach-think-s16"), ("e19", "teach-think-s17")], False),
        ("judging", [("e16", "judge-s15"), ("e19", "judge-s16"), ("e19", "judge-s17")], True)]
for ax, (name, runs, dashed) in zip(axes, arms):
    ax.axhline(score(PROBE[f"base/{key}"]), color=GREY, linewidth=0.8, linestyle=(0, (1, 1.5)))
    for c in seeds(runs, key):
        ax.plot(X, c, color=BLUE, linewidth=0.9, linestyle=(0, (4, 2)) if dashed else "-")
    ax.set_title(name, fontsize=7.5, color=INK, loc="left", pad=2)
    ax.set_ylim(0, 80)
    ax.set_xlim(0, 1200)
    ax.set_xticks([0, 600, 1200])
    ax.set_xlabel("practice tasks", fontsize=7)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
axes[0].set_ylabel("probe solved (%)", fontsize=7)
fig.tight_layout(w_pad=0.4)
fig.savefig(OUT / "reasoning_curves.pdf", bbox_inches="tight")
plt.close(fig)

# ---- Figure 4: side effects outside multi-turn, split into calling and abstaining (seed 15; reasoning kept seed 15) ----
# BFCL's live and non-live groups mix questions that need a call with questions where no tool fits (irrelevance, where
# the right answer is no call). They move in opposite directions under LoRA, so they are shown apart. Each cell is a
# paired test against the floor on the same questions (McNemar's exact test), Holm-corrected across the figure, as in
# E20 (lab/e20/stats.py).
from math import comb

BFCL = json.loads((DATA / "per_question/bfcl.json").read_text())["exams"]
GROUPS = [("calling", "calling", ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
                                  "parallel_multiple", "live_simple", "live_multiple", "live_parallel",
                                  "live_parallel_multiple", "live_relevance"]),
          ("abstaining", "abstaining", ["irrelevance", "live_irrelevance"]),
          ("memory", "memory", ["memory_kv", "memory_vector", "memory_rec_sum"])]
ROWS = [("notes T", "e15", "teach-s15"), ("notes J", "e15", "judge-s15"), ("retrieval T", "e17", "teach-s15"),
        ("retrieval J", "e17", "judge-s15"), ("LoRA T", "e16", "teach-s15"), ("LoRA J", "e16", "judge-s15"),
        ("reasoning kept", "e19", "teach-think-s15")]


def mcnemar(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


cells = {}
for ri, (_, exp, run) in enumerate(ROWS):
    for mi, (m, _) in enumerate(MODELS):
        ex = BFCL.get(f"{exp}-{m}-{run}")
        if ex is None:
            continue
        fl = BFCL[f"floor-{m}"]
        for gi, (g, _, cats) in enumerate(GROUPS):
            a = "".join(fl[c] for c in cats)
            bb = "".join(ex[c] for c in cats)
            gained = sum(x == "0" and y == "1" for x, y in zip(a, bb))
            lost = sum(x == "1" and y == "0" for x, y in zip(a, bb))
            cells[(ri, mi, gi)] = {"diff": 100 * (bb.count("1") - a.count("1")) / len(a), "p": mcnemar(gained, lost),
                                   "testable": gained + lost > 0, "n": len(a)}
tested = sorted((c["p"], k) for k, c in cells.items() if c["testable"])
m_tests, running = len(tested), 0.0
for i, (pv, k) in enumerate(tested):  # Holm step-down
    running = max(running, min(1.0, (m_tests - i) * pv))
    cells[k]["p_holm"] = running
for c in cells.values():
    c["significant"] = c.get("p_holm", 1.0) < 0.05
json.dump({f"{ROWS[r][0]}|{MODELS[mi][0]}|{GROUPS[g][0]}": v for (r, mi, g), v in cells.items()},
          open(DATA / "side_effects.json", "w"), indent=1)

cols = [(mi, gi) for mi in range(len(MODELS)) for gi in range(len(GROUPS))]
grid = [[cells[(ri, mi, gi)]["diff"] if (ri, mi, gi) in cells else float("nan") for mi, gi in cols]
        for ri in range(len(ROWS))]
sig = [[(ri, mi, gi) in cells and cells[(ri, mi, gi)]["significant"] for mi, gi in cols] for ri in range(len(ROWS))]
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
cmap.set_bad("#ffffff")
lim = 60
fig, ax = plt.subplots(figsize=(7.2, 2.6))
im = ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        if v != v:  # not examined
            ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=6.5)
            continue
        txt = f"{v:+.0f}" if abs(v) >= 0.5 else "0"
        ax.text(j, i, txt + ("*" if sig[i][j] else ""), ha="center", va="center", fontsize=6.3,
                color="#ffffff" if abs(v) > 0.6 * lim else INK, fontweight="bold" if sig[i][j] else "normal")
ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS])
ax.set_xticks(range(len(cols)), [GROUPS[gi][1] for _, gi in cols], rotation=90, fontsize=6.3)
for k, (_, label) in enumerate(MODELS):
    ax.text(k * 3 + 1, -0.85, label, ha="center", va="bottom", fontsize=7, color=INK)
    if k:
        ax.axvline(k * 3 - 0.5, color="#ffffff", linewidth=2.5)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("change against frozen score (points)", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "side_effects.pdf", bbox_inches="tight")
plt.close(fig)
# ---- Over-calling (Figure 5): (a) LoRA teaching, six models; (b) Qwen3-1.7B, the three LoRA arms (seed 15) ----
# The full grid (every run, calling / abstaining / memory, with tests) is side_effects.json, written above.
DARK, LIGHT = "#1c5cab", "#86b6ef"  # one hue, two steps (validated as an ordinal pair)


def pair_bars(ax, groups, labels, wd=0.36):
    """groups: list of (row index, model index); one pair of bars (calls, abstention) per group."""
    xs = list(range(len(groups)))
    for off, gi, color, name in ((-wd / 2, 0, DARK, "correct calls"), (wd / 2, 1, LIGHT, "correct abstention")):
        vals = [cells[(r, mi, gi)]["diff"] for r, mi in groups]
        sigs = [cells[(r, mi, gi)]["significant"] for r, mi in groups]
        ax.bar([x + off for x in xs], vals, width=wd, color=color, label=name, edgecolor="#fcfcfb", linewidth=0.6)
        for x, v, sg in zip(xs, vals, sigs):
            ax.text(x + off, v + (1.5 if v >= 0 else -1.5), f"{v:+.0f}" + ("*" if sg else ""), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=5.0, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ax.set_xticks(xs, labels, rotation=30, ha="right", fontsize=6.5)
    ax.set_ylim(-88, 45)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)


fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True, gridspec_kw={"width_ratios": [6, 3.4]})
pair_bars(axa, [(4, mi) for mi in range(len(MODELS))], [m[1] for m in MODELS])
axa.set_title("(a) LoRA, oracle teaching", fontsize=7.5, color=INK, loc="left", pad=3)
axa.set_ylabel("change against frozen score (points)", fontsize=7)
pair_bars(axb, [(4, 0), (6, 0), (5, 0)], ["teaching", "reasoning kept", "judging"])
axb.set_title("(b) Qwen3-1.7B, three LoRA arms", fontsize=7.5, color=INK, loc="left", pad=3)
h, l = axa.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.07), handlelength=1.2)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / "overcalling.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", *(p.name for p in sorted(OUT.glob("*.pdf"))))

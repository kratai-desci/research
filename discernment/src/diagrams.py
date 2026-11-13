"""Draw the paper's two diagrams (Fig 1: the creature; Fig 9: how the three versions of the sense decide) into paper/figures/.

  uv run --with matplotlib python src/diagrams.py [out dir]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EDGE, ACC, INK, QUIET, TINT = "#9a9a90", "#2a6fd6", "#1d1d1d", "#5f5f5f", "#efefeb"
plt.rcParams.update({"font.family": "sans-serif", "savefig.bbox": "tight"})


def canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title=None, lines=(), main=False, fill=None, dashed=False, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", lw=2 if main else 1.1,
                                ec=ACC if main else EDGE, fc=(fill or ("#e4edfb" if main else "white")), ls="--" if dashed else "-"))
    ty = y + 22
    if title:
        ax.text(x + 12, ty, title, fontsize=size, fontweight="bold", color=INK, va="center"); ty += 19
    for line in lines:
        ax.text(x + 12, ty, line, fontsize=size - 1.2, color=INK if main else QUIET, va="center"); ty += 16


def arrow(ax, pts, label=None, at=None):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        last = (x2, y2) == pts[-1]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>" if last else "-", mutation_scale=10, color=EDGE, lw=1.1, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*at, label, fontsize=8.5, color=QUIET, va="center")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig); print(OUT / f"{name}.pdf")


# Fig 1: the creature
fig, ax = canvas(760, 720)
ax.text(24, 30, "A judge decides what the creature learns each night", fontsize=12, fontweight="bold", color=INK, va="center")
box(ax, 24, 56, 712, 132, fill=TINT); ax.text(40, 78, "The creature", fontsize=10.5, fontweight="bold", color=INK, va="center")
box(ax, 40, 100, 520, 72, "Learning adapter: a LoRA (rank 8, top 8 layers)", ["everything the creature learns lives here;", "loaded into the model each morning"])
box(ax, 576, 100, 144, 72, "Base model", ["Qwen3 0.6B / 1.7B", "frozen, never changes"])
bw, rowY, bh = 154, 272, 104
cx = lambda i: 24 + i * (bw + 32); mid = lambda i: cx(i) + bw / 2; cy = rowY + bh / 2
steps = [("1  Day: work", ["30 new tasks;", "explores, temp. 0.7;", "the user reacts"], False),
         ("2  Judge", ["picks the attempts", "worth learning:", "oracle, stated verdict,", "self, both or sense"], True),
         ("3  Night: sleep", ["SFT + DPO on the", "lessons; replays", "the last 150"], False),
         ("4  Exam", ["held-out tasks,", "never learned from;", "greedy, temp. 0"], False)]
for i, (t, lines, main) in enumerate(steps):
    box(ax, cx(i), rowY, bw, bh, t, lines, main=main)
for i in range(3):
    arrow(ax, [(cx(i) + bw, cy), (cx(i + 1), cy)])
arrow(ax, [(mid(0), 172), (mid(0), rowY)], "runs the day's work", (mid(0) + 8, 228))
arrow(ax, [(mid(2), rowY), (mid(2), 172)], "updates the adapter", (mid(2) + 8, 228))
arrow(ax, [(mid(1), 408), (mid(1), rowY + bh)], "sense judge only", (mid(1) + 8, 396))
arrow(ax, [(mid(3), rowY + bh), (mid(3), 536), (mid(0), 536), (mid(0), rowY + bh)])
ax.text((mid(0) + mid(3)) / 2, 552, "next day", fontsize=8.5, color=QUIET, ha="center", va="center")
box(ax, 186, 408, 334, 104, "Feedback sense: separate and frozen",
    ["Qwen3-0.6B with its own LoRA, never trained", "during a life. Reads the request, the user's", "words and next move, never the result.",
     "v1, v2: one question; v3 first asks if clear."])
arrow(ax, [(470, 592), (470, 512)], "frozen result", (478, 578))
box(ax, 24, 592, 712, 116, dashed=True)
ax.text(40, 612, "Building the sense, before any life (3.2): v2 and v3; v1 used WildFB's labels", fontsize=10, fontweight="bold", color=INK, va="center")
build = [("Reactions", ["110,906 from 6 datasets", "+ hard cases, written"]), ("Teacher: Qwen3-32B", ["labels each reaction", "with its probabilities"]),
         ("Student: Qwen3-0.6B", ["a LoRA learns to match", "the teacher (KL)"]), ("Calibration", ["threshold: at most 3%", "of noes read as a yes"])]
for i, (t, lines) in enumerate(build):
    box(ax, 40 + 176 * i, 626, 152, 70, t, lines, size=8.2)
    if i < 3:
        arrow(ax, [(40 + 176 * i + 152, 661), (40 + 176 * (i + 1), 661)])
save(fig, "fig1_creature")

# Fig 9: how the three versions of the sense decide
fig, ax = canvas(760, 520)
ax.text(24, 28, "v1 and v2 ask which score; v3 first asks whether there is a verdict at all", fontsize=12, fontweight="bold", color=INK, va="center")
rows = [(68, "Sense v1 (Experiment 2): learned WildFB's own labels", "null when it is less than 0.70 sure (a fixed threshold)"),
        (198, "Sense v2 (Experiment 3): learned a teacher's labels", "null when it is less than 0.65 sure (calibrated)")]
for y, title, rule in rows:
    ax.text(24, y, title, fontsize=10.5, fontweight="bold", color=INK, va="center")
    box(ax, 24, y + 14, 120, 76, "Reaction", ["request, next", "move, words"])
    box(ax, 168, y + 14, 372, 76, "One question: which score?", ["100, 80, 20, 0 or null, in a single step", rule])
    arrow(ax, [(144, y + 52), (168, y + 52)]); arrow(ax, [(540, y + 52), (564, y + 52)])
ax.text(24, 328, "Sense v3 (Experiments 4 and 5): learned a teacher's two answers", fontsize=10.5, fontweight="bold", color=INK, va="center")
box(ax, 24, 342, 120, 76, "Reaction", ["request, next", "move, words"])
box(ax, 168, 342, 172, 76, "1  A clear verdict?", ["on the result, not the", "effort; at least 0.80 sure"], main=True)
box(ax, 368, 342, 172, 76, "2  Which score?", ["100, 80, 20 or 0;", "at least 0.70 sure"])
box(ax, 168, 450, 172, 44, "null: nothing learned")
ax.text(356, 464, "stops here: thanks for trying, the next request with", fontsize=8.5, color=QUIET, va="center")
ax.text(356, 480, "no comment, sarcasm, settling, silence (Table 2)", fontsize=8.5, color=QUIET, va="center")
box(ax, 564, 82, 172, 336, "The creature then")
for y, (a, b) in zip((190, 250, 310), (("yes (100, 80): a lesson", "80 at half weight"), ("no (20, 0): a retry", "on WebShop, with a hint"), ("null: nothing", ""))):
    ax.text(578, y, a, fontsize=9, color=INK, va="center"); ax.text(578, y + 16, b, fontsize=8.5, color=QUIET, va="center")
for pts in ([(144, 380), (168, 380)], [(340, 380), (368, 380)], [(540, 380), (564, 380)], [(254, 418), (254, 450)]):
    arrow(ax, pts)
ax.text(343, 370, "yes", fontsize=8, color=QUIET); ax.text(262, 436, "no", fontsize=8, color=QUIET)
save(fig, "fig9_sense_decision")

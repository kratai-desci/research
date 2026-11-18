"""Draw the paper's two diagrams (Fig 1: the creature; Fig 9: how sense v3 decides) into paper/figures/.

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


# Fig 1: the creature (wide and short, so it sits in the text instead of taking a page)
fig, ax = canvas(760, 372)
box(ax, 24, 6, 712, 94, fill=TINT); ax.text(40, 22, "The creature", fontsize=10.5, fontweight="bold", color=INK, va="center")
box(ax, 40, 34, 520, 58, "Learning adapter: a LoRA (rank 8, top 8 layers)", ["everything the creature learns lives here; loaded each morning"])
box(ax, 576, 34, 144, 58, "Base model", ["Qwen3 0.6B or 1.7B"])
bw, rowY, bh = 154, 136, 100
cx = lambda i: 24 + i * (bw + 32); mid = lambda i: cx(i) + bw / 2; cy = rowY + bh / 2
steps = [("1  Day: work", ["30 new tasks;", "explores, temp. 0.7;", "the user reacts"], False),
         ("2  Judge", ["picks the attempts", "worth learning:", "oracle, stated verdict,", "self, both or sense"], True),
         ("3  Night: sleep", ["SFT + DPO on the", "lessons; replays", "the last 150"], False),
         ("4  Exam", ["held-out tasks,", "never learned from;", "greedy, temp. 0"], False)]
for i, (t, lines, main) in enumerate(steps):
    box(ax, cx(i), rowY, bw, bh, t, lines, main=main)
for i in range(3):
    arrow(ax, [(cx(i) + bw, cy), (cx(i + 1), cy)])
arrow(ax, [(mid(0), 92), (mid(0), rowY)], "runs the day's work", (mid(0) + 8, 114))
arrow(ax, [(mid(2), rowY), (mid(2), 92)], "updates the adapter", (mid(2) + 8, 114))
box(ax, 24, 266, 320, 100, "Feedback sense: separate and frozen",
    ["Qwen3-0.6B with its own LoRA, never trained", "during a life. Reads the request, the user's", "words and next move, never the result.",
     "v1, v2: one question; v3 first asks if clear."])
arrow(ax, [(mid(1), 266), (mid(1), rowY + bh)], "sense judge only", (mid(1) + 8, 251))
box(ax, 372, 266, 364, 100, "Building the sense (v2, v3), before any life", [
    "Reactions: 110,906 from 6 datasets + hard cases", "Teacher Qwen3-32B labels each, with probabilities",
    "Student Qwen3-0.6B (LoRA) learns to match (KL)", "Calibrated: at most 3% of noes read as a yes"], dashed=True)
arrow(ax, [(372, 316), (344, 316)])
save(fig, "fig1_creature")

# Fig 9: how sense v3 decides (v1 and v2 ask only the second question)
fig, ax = canvas(760, 262)
ax.text(24, 28, "Sense v3 asks two questions before the creature learns anything", fontsize=12, fontweight="bold", color=INK, va="center")
box(ax, 24, 56, 120, 76, "Reaction", ["request, next", "move, words"])
box(ax, 168, 56, 172, 76, "1  A clear verdict?", ["on the result, not the", "effort; at least 0.80 sure"], main=True)
box(ax, 368, 56, 172, 76, "2  Which score?", ["100, 80, 20 or 0;", "at least 0.70 sure"])
box(ax, 168, 168, 172, 44, "null: nothing learned")
for k, line in enumerate(["stops here: thanks for trying,", "the next request with no comment,", "sarcasm, settling, silence (Table 2)"]):
    ax.text(356, 174 + 16 * k, line, fontsize=8.5, color=QUIET, va="center")
box(ax, 564, 56, 172, 180, "The creature then")
for y, (a, b) in zip((100, 148, 196), (("yes (100, 80): a lesson", "80 at half weight"), ("no (20, 0): a retry", "on WebShop, with a hint"), ("null: nothing", ""))):
    ax.text(578, y, a, fontsize=9, color=INK, va="center"); ax.text(578, y + 16, b, fontsize=8.5, color=QUIET, va="center")
for pts in ([(144, 94), (168, 94)], [(340, 94), (368, 94)], [(540, 94), (564, 94)], [(254, 132), (254, 168)]):
    arrow(ax, pts)
ax.text(343, 84, "yes", fontsize=8, color=QUIET); ax.text(262, 152, "no", fontsize=8, color=QUIET)
save(fig, "fig9_sense_decision")

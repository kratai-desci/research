"""Sense v2 student: Qwen3-0.6B + LoRA learns the teacher's five-score likelihoods (soft labels), then is calibrated.

Differences from v1 (train.py): targets are the teacher's distribution, not one label; rows are balanced by task
type and the teacher's best score; the request is blanked in half the training rows so the sense relies on the
reaction; one temperature is fitted on validation so stated confidence means what it says; and the threshold is
the lowest confidence that keeps false accepts (a clear no read as a yes) under --false-accept on validation.

--hold-out <task> leaves one task type out of training and reports on it: the leave-one-task-out test.

  uv run python -m lab.sense.train2 --data lab/data/sense2 --out lab/data/sense2/adapter
  uv run python -m lab.sense.train2 --data lab/data/sense2 --hold-out coding --out lab/data/sense2/adapter-no-coding
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.sense import LABELS
from lab.sense.status import Status
from lab.sense.train import ALPHA, MODEL, MODULES, RANK, Scorer, load

PER_CELL = 1500  # rows kept per (task type, teacher's best score)
YES, NO = {"80", "100"}, {"0", "20"}
side = lambda lab: "yes" if lab in YES else "no" if lab in NO else "null"


def rows_from(data: Path) -> list[dict]:
    rows = []
    for name in ("scored.jsonl", "gaps.jsonl"):
        if (data / name).exists():
            for line in (data / name).open():
                r = json.loads(line)
                if r.get("teacher"):
                    rows.append({"request": r["request"], "next_move": r.get("next_move") or ("replied" if r["said"] else "said nothing"),
                                 "said": r["said"], "task": r.get("task") or "other", "source": r["source"],
                                 "target": [r["teacher"][lab] for lab in LABELS]})
    return rows


def balance(rows: list[dict], rng: random.Random) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["task"], max(range(5), key=r["target"].__getitem__))].append(r)
    return [r for cell in cells.values() for r in rng.sample(cell, min(len(cell), PER_CELL))]


@torch.no_grad()
def logits_of(model, scorer: Scorer, rows: list[dict], batch: int = 32) -> torch.Tensor:
    model.eval()
    out = torch.cat([scorer.logits(model, rows[i:i + batch]).cpu() for i in range(0, len(rows), batch)])
    model.train()
    return out


def fit_temperature(logits: torch.Tensor, target: torch.Tensor) -> float:
    """The single temperature that best matches the teacher's distribution on validation."""
    best = min((t / 20 for t in range(10, 81)), key=lambda t: F.kl_div(F.log_softmax(logits / t, -1), target, reduction="batchmean").item())
    return round(best, 2)


def report(rows: list[dict], probs: torch.Tensor, threshold: float) -> dict:
    """Against the teacher's best guess: side agreement, false accepts, coverage, calibration error."""
    conf, pred = probs.max(-1)
    said = [LABELS[p] if c >= threshold else "null" for p, c in zip(pred.tolist(), conf.tolist())]
    truth = [LABELS[max(range(5), key=r["target"].__getitem__)] for r in rows]
    answered = [(s, g) for s, g in zip(said, truth) if s != "null"]
    noes = [s for s, g in zip(said, truth) if g in NO]
    bins = defaultdict(list)  # expected calibration error over 10 confidence bins, against the teacher's best guess
    for c, p, g in zip(conf.tolist(), pred.tolist(), truth):
        bins[min(9, int(c * 10))].append((c, LABELS[p] == g))
    ece = sum(len(b) * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b)) for b in bins.values()) / len(truth)
    return {"n": len(rows), "coverage": round(len(answered) / len(rows), 3),
            "side_agreement": round(sum(side(s) == side(g) for s, g in answered) / max(1, len(answered)), 3),
            "false_accept": round(sum(s in YES for s in noes) / max(1, len(noes)), 3),
            "exact": round(sum(s == g for s, g in zip(said, truth)) / len(rows), 3), "ece": round(ece, 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("lab/data/sense2"))
    p.add_argument("--out", type=Path, default=Path("lab/data/sense2/adapter"))
    p.add_argument("--hold-out", default="", help="a task type left out of training and tested on")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--micro", type=int, default=8, help="rows per forward pass; gradients add up to --batch (fits an L4)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--false-accept", type=float, default=0.03)
    p.add_argument("--shopping", type=Path, default=Path("lab/data/sense/test_shopping.jsonl"), help="v1's unseen-domain test")
    args = p.parse_args()
    tag = f"-no-{args.hold_out.replace(' ', '-')}" if args.hold_out else ""
    status = Status(args.data / f"train{tag}-status.json", remote=f"sense2/train{tag}-status.json")
    rng = random.Random(0)
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = rows_from(args.data)
    held = [r for r in rows if r["task"] == args.hold_out] if args.hold_out else []
    rows = balance([r for r in rows if r["task"] != args.hold_out], rng)
    rng.shuffle(rows)
    val, train = rows[:2000], rows[2000:]
    held = rng.sample(held, min(2000, len(held)))
    shop = load(args.shopping, 2000) if args.shopping.exists() else []
    status.update(stage="training", hold_out=args.hold_out or None, data={"train": len(train), "val": len(val), "held_out": len(held)},
                  force=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    scorer = Scorer(tok, dev)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    model = get_peft_model(base, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05, target_modules=MODULES, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    steps = math.ceil(len(train) * args.epochs / args.batch)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / steps))
    status.update(train={"rows": len(train), "steps": steps, "step": 0, "loss": [],
                         "trainable_params": sum(q.numel() for q in model.parameters() if q.requires_grad)}, force=True)

    order, losses, t0 = [], [], time.time()
    for step in range(steps):
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        batch = [dict(train[order.pop()]) for _ in range(min(args.batch, len(order)))]
        for r in batch:  # half the time the sense must read the reaction without the request
            if rng.random() < 0.5:
                r["request"] = "(not shown)"
        opt.zero_grad()
        loss = 0.0
        for i in range(0, len(batch), args.micro):  # same update as one batch, in slices that fit in memory
            part = batch[i:i + args.micro]
            target = torch.tensor([r["target"] for r in part], device=dev)
            part_loss = F.kl_div(F.log_softmax(scorer.logits(model, part), -1), target, reduction="sum") / len(batch)
            part_loss.backward()
            loss += part_loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if (step + 1) % 25 == 0 or step + 1 == steps:
            per = (time.time() - t0) / (step + 1)
            status.update(train={"step": step + 1, "loss": status.state["train"]["loss"] + [[step + 1, round(sum(losses[-25:]) / len(losses[-25:]), 4)]],
                                 "eta_s": round(per * (steps - step - 1))})

    status.update(stage="calibrating", force=True)
    val_logits = logits_of(model, scorer, val)
    temp = fit_temperature(val_logits, torch.tensor([r["target"] for r in val]))
    val_probs = F.softmax(val_logits / temp, -1)
    # The lowest confidence threshold whose false accepts on validation stay under the target.
    threshold = next((t / 100 for t in range(30, 100) if report(val, val_probs, t / 100)["false_accept"] <= args.false_accept), 0.99)
    evals = {"val": report(val, val_probs, threshold)}
    if held:
        evals["held_out_task"] = report(held, F.softmax(logits_of(model, scorer, held) / temp, -1), threshold)
    if shop:  # v1's shopping test: labels from the template user's words, never seen in training
        sp = F.softmax(logits_of(model, scorer, shop) / temp, -1)
        evals["shopping"] = report([{**r, "target": [float(r["label"] == lab) for lab in LABELS]} for r in shop], sp, threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    (args.out / "calibration.json").write_text(json.dumps({"temperature": temp, "threshold": threshold}))
    status.update(stage="done", calibration={"temperature": temp, "threshold": threshold}, eval2=evals, force=True)
    print(json.dumps({"temperature": temp, "threshold": threshold, **evals}, indent=2))


if __name__ == "__main__":
    main()

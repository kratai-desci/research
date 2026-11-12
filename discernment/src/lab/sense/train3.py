"""Sense v3 student: Qwen3-0.6B + one LoRA learns the teacher's answers to v3's two questions, then is calibrated.

Question 1 (clear verdict? yes / no) is learned on every row; question 2 (score 0 / 20 / 80 / 100) is learned in
proportion to how clear the teacher found the reaction. Both are soft labels (KL to the teacher). As in v2, rows
are balanced by task type and the teacher's answer, and the request is blanked half the time.

Reading: null unless P(clear) >= the clear threshold; then the most likely score, null if its probability is under
the score threshold. The two thresholds are the pair that keeps false accepts (a teacher's clear no, or a reaction
the teacher found unclear, read as a yes) at or under --false-accept on validation, with the most yes-answers kept.

--live tests the student on reactions from our own runs (E6, E8: lab/sense/live.py), never trained on; --compare
scores the same reactions with a v2 sense for comparison.

  uv run python -m lab.sense.train3 --data lab/data/sense3 --out lab/data/sense3/adapter --live lab/data/sense3/live.jsonl
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
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.sense import CLEAR, LABELS, SCORES, prompt, prompt3
from lab.sense.status import Status
from lab.sense.teach3 import label
from lab.sense.train import ALPHA, MAX_LEN, MODEL, MODULES, RANK

PER_CELL = 1500  # rows kept per (task type, teacher's answer)
YES = {"80", "100"}


class Scorer:
    """Turns rows into one question's prompt ids and reads that question's answer logits."""

    def __init__(self, tok, dev: str, step: str):
        self.tok, self.dev, self.step = tok, dev, step
        labels = CLEAR if step == "clear" else SCORES
        self.ids_ = [tok(lab, add_special_tokens=False).input_ids[0] for lab in labels]
        assert len(set(self.ids_)) == len(labels), "answer first tokens must differ"
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def ids(self, row: dict) -> list[int]:
        msgs = prompt3(row, self.step) if self.step in ("clear", "score") else prompt(row)
        ids = self.tok(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                       add_special_tokens=False).input_ids
        return ids if len(ids) <= MAX_LEN else ids[:MAX_LEN // 2] + ids[-MAX_LEN // 2:]

    def logits(self, model, rows: list[dict]) -> torch.Tensor:
        seqs = [self.ids(r) for r in rows]
        n = max(map(len, seqs))
        ids = torch.tensor([s + [self.pad] * (n - len(s)) for s in seqs], device=self.dev)
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=self.dev)
        last = torch.tensor([len(s) - 1 for s in seqs], device=self.dev)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        hidden = base.model(input_ids=ids, attention_mask=mask).last_hidden_state
        return base.lm_head(hidden[torch.arange(len(seqs), device=self.dev), last]).float()[:, self.ids_]


def rows_from(path: Path) -> list[dict]:
    rows = []
    for line in path.open():
        r = json.loads(line)
        if r.get("teacher3"):
            t = r["teacher3"]
            rows.append({"request": r["request"], "said": r["said"], "task": r.get("task") or "other", "source": r["source"],
                         "next_move": r.get("next_move") or ("replied" if r["said"] else "said nothing"),
                         "clear": [t["clear"], 1 - t["clear"]], "score": [t["score"][s] for s in SCORES], "truth": label(t)})
    return rows


@torch.no_grad()
def probs_of(model, scorer: Scorer, rows: list[dict], temp: float = 1.0, batch: int = 32) -> torch.Tensor:
    model.eval()
    out = torch.cat([F.softmax(scorer.logits(model, rows[i:i + batch]).cpu() / temp, -1) for i in range(0, len(rows), batch)])
    model.train()
    return out


def fit_temperature(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> float:
    def loss(t):
        kl = F.kl_div(F.log_softmax(logits / t, -1), target, reduction="none").sum(-1)
        return float((kl * weight).sum() / weight.sum() if weight is not None else kl.mean())
    return round(min((t / 20 for t in range(10, 81)), key=loss), 2)


def read(pc: torch.Tensor, ps: torch.Tensor, t_clear: float, t_score: float) -> list[str]:
    """v3's reading: null unless clear enough; then the best score, null if it is not confident enough."""
    out = []
    for c, s in zip(pc.tolist(), ps.tolist()):
        best = max(range(4), key=s.__getitem__)
        out.append(SCORES[best] if c[0] >= t_clear and s[best] >= t_score else "null")
    return out


def report(truth: list[str], said: list[str]) -> dict:
    """Against the teacher: false accepts (a no or an unclear reaction read as a yes), yes kept, side agreement."""
    side = lambda lab: "yes" if lab in YES else "no" if lab in ("0", "20") else "null"
    not_yes = [s for s, g in zip(said, truth) if g not in YES]
    yes = [s for s, g in zip(said, truth) if g in YES]
    answered = [(s, g) for s, g in zip(said, truth) if s != "null" and g != "null"]
    return {"n": len(truth), "false_accept": round(sum(s in YES for s in not_yes) / max(1, len(not_yes)), 3),
            "yes_kept": round(sum(s in YES for s in yes) / max(1, len(yes)), 3),
            "null_agreement": round(sum((s == "null") == (g == "null") for s, g in zip(said, truth)) / len(truth), 3),
            "side_agreement": round(sum(side(s) == side(g) for s, g in answered) / max(1, len(answered)), 3),
            "exact": round(sum(s == g for s, g in zip(said, truth)) / len(truth), 3)}


def live_report(rows: list[dict], said: list[str]) -> dict:
    """On our own runs' reactions: which behaviours were read as a yes. A yes is right only for praise and accept."""
    by = defaultdict(lambda: [0, 0])
    for r, s in zip(rows, said):
        by[r["kind"]][0] += 1
        by[r["kind"]][1] += s in YES
    should = {"praise", "accept"}
    wrong = [s in YES for r, s in zip(rows, said) if r["kind"] not in should]
    right = [s in YES for r, s in zip(rows, said) if r["kind"] in should]
    gave_up = [s in YES for r, s in zip(rows, said) if r["kind"] in ("give_up", "polite_unhappy")]
    return {"n": len(rows), "false_accept": round(sum(wrong) / max(1, len(wrong)), 3),
            "give_up_read_as_yes": round(sum(gave_up) / max(1, len(gave_up)), 3),
            "accept_read_as_yes": round(sum(right) / max(1, len(right)), 3),
            "yes_by_behaviour": {k: f"{v[1]}/{v[0]}" for k, v in sorted(by.items())}}


def compare_v2(path: Path, rows: list[dict], tok, dev: str) -> dict:
    """The v2 sense on the same reactions, read with its own calibration."""
    from lab.sense.train import Scorer as V2Scorer

    cal = json.loads((path / "calibration.json").read_text())
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev), str(path))
    p = probs_of(model, V2Scorer(tok, dev), rows, cal["temperature"])
    conf, pred = p.max(-1)
    said = [LABELS[i] if c >= cal["threshold"] else "null" for i, c in zip(pred.tolist(), conf.tolist())]
    del model
    return live_report(rows, said)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("lab/data/sense3"))
    p.add_argument("--out", type=Path, default=Path("lab/data/sense3/adapter"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--false-accept", type=float, default=0.03)
    p.add_argument("--live", type=Path, help="reactions from our runs, with the behaviour the user drew (lab/sense/live.py)")
    p.add_argument("--compare", type=Path, help="a v2 sense adapter to score the live reactions too")
    args = p.parse_args()
    status = Status(args.data / "train-status.json", remote="sense3/train-status.json")
    rng = random.Random(0)
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = rows_from(args.data / "scored.jsonl")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["task"], r["truth"])].append(r)
    rows = [r for cell in cells.values() for r in rng.sample(cell, min(len(cell), PER_CELL))]
    rng.shuffle(rows)
    val, train = rows[:2000], rows[2000:]
    live = [json.loads(line) for line in args.live.open()] if args.live and args.live.exists() else []
    status.update(stage="training", data={"train": len(train), "val": len(val), "live": len(live),
                                          "labels": {k: sum(r["truth"] == k for r in rows) for k in SCORES + ["null"]}}, force=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    clear_s, score_s = Scorer(tok, dev, "clear"), Scorer(tok, dev, "score")
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    model = get_peft_model(base, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05, target_modules=MODULES, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    steps = math.ceil(len(train) * args.epochs / args.batch)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / steps))
    status.update(train={"rows": len(train), "steps": steps, "step": 0, "loss": []}, force=True)

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
        for i in range(0, len(batch), args.micro):
            part = batch[i:i + args.micro]
            tc = torch.tensor([r["clear"] for r in part], device=dev)
            ts = torch.tensor([r["score"] for r in part], device=dev)
            kl_c = F.kl_div(F.log_softmax(clear_s.logits(model, part), -1), tc, reduction="none").sum(-1)
            kl_s = F.kl_div(F.log_softmax(score_s.logits(model, part), -1), ts, reduction="none").sum(-1)
            part_loss = (kl_c + tc[:, 0] * kl_s).sum() / len(batch)  # the score counts as much as the reaction is clear
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
    with torch.no_grad():
        model.eval()
        lc = torch.cat([clear_s.logits(model, val[i:i + 32]).cpu() for i in range(0, len(val), 32)])
        ls = torch.cat([score_s.logits(model, val[i:i + 32]).cpu() for i in range(0, len(val), 32)])
    tc = fit_temperature(lc, torch.tensor([r["clear"] for r in val]))
    ts = fit_temperature(ls, torch.tensor([r["score"] for r in val]), torch.tensor([r["clear"][0] for r in val]))
    pc, ps = F.softmax(lc / tc, -1), F.softmax(ls / ts, -1)
    truth = [r["truth"] for r in val]
    best = None  # the pair of thresholds that keeps false accepts under the target with the most yes kept
    for a in range(30, 100, 5):
        for b in range(30, 100, 5):
            rep = report(truth, read(pc, ps, a / 100, b / 100))
            if rep["false_accept"] <= args.false_accept and (best is None or rep["yes_kept"] > best[2]["yes_kept"]):
                best = (a / 100, b / 100, rep)
    t_clear, t_score, val_rep = best or (0.95, 0.95, report(truth, read(pc, ps, 0.95, 0.95)))
    cal = {"version": 3, "temperature_clear": tc, "temperature_score": ts, "threshold_clear": t_clear, "threshold_score": t_score}
    evals = {"val": val_rep}
    if live:
        lp_c, lp_s = probs_of(model, clear_s, live, tc), probs_of(model, score_s, live, ts)
        evals["live"] = live_report(live, read(lp_c, lp_s, t_clear, t_score))
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    (args.out / "calibration.json").write_text(json.dumps(cal))
    if live and args.compare and args.compare.exists():
        del model
        torch.cuda.empty_cache()
        evals["live_v2"] = compare_v2(args.compare, live, tok, dev)
    status.update(stage="done", calibration=cal, eval3=evals, force=True)
    print(json.dumps({**cal, **evals}, indent=2))


if __name__ == "__main__":
    main()

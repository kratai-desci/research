"""Train the feedback sense: a LoRA on Qwen3-0.6B that answers with one score label, then freeze it.

The model reads the prompt from lab.sense.prompt(); its answer is the first token of one of the five labels
(0, 20, 80, 100, null), so scoring is one forward pass: the five label tokens' logits, softmaxed, give the label
and its confidence. Training is cross-entropy over those five logits. Evaluation runs on the plain model first
(zero-shot) and again after training, on held-out WildFB, unclear replies and the unseen shopping domain.

  uv run python -m lab.sense.train --data lab/data/sense --out lab/data/sense/adapter
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.sense import LABELS, prompt
from lab.sense.status import Status

MODEL = "Qwen/Qwen3-0.6B"
RANK, ALPHA = 16, 32
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 448
YES, NO = {"80", "100"}, {"0", "20"}
THRESHOLDS = [0.0, 0.5, 0.7, 0.9]  # below this confidence the answer counts as null


def load(path: Path, limit: int = 0, seed: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return rows


class Scorer:
    """Turns rows into prompt ids and reads the five label logits at the answer position."""

    def __init__(self, tok, dev: str):
        self.tok, self.dev = tok, dev
        self.label_ids = [tok(lab, add_special_tokens=False).input_ids[0] for lab in LABELS]
        assert len(set(self.label_ids)) == len(LABELS), "label first tokens must differ"
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def ids(self, row: dict) -> list[int]:
        text = self.tok.apply_chat_template(prompt(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(text, add_special_tokens=False).input_ids
        return ids if len(ids) <= MAX_LEN else ids[:MAX_LEN // 2] + ids[-MAX_LEN // 2:]

    def logits(self, model, rows: list[dict]) -> torch.Tensor:
        seqs = [self.ids(r) for r in rows]
        n = max(map(len, seqs))
        ids = torch.tensor([s + [self.pad] * (n - len(s)) for s in seqs], device=self.dev)
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=self.dev)
        last = torch.tensor([len(s) - 1 for s in seqs], device=self.dev)
        hidden = model.get_base_model().model(input_ids=ids, attention_mask=mask).last_hidden_state if hasattr(
            model, "get_base_model") else model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        at = hidden[torch.arange(len(seqs), device=self.dev), last]
        head = model.get_base_model().lm_head if hasattr(model, "get_base_model") else model.lm_head
        return head(at).float()[:, self.label_ids]


def metrics(rows: list[dict], probs: torch.Tensor) -> dict:
    """Accuracy, false accepts and abstaining, at several confidence thresholds."""
    conf, pred = probs.max(-1)
    out = {"n": len(rows), "confusion": {}}
    for t in THRESHOLDS:
        said = [LABELS[p] if c >= t else "null" for p, c in zip(pred.tolist(), conf.tolist())]
        truth = [r["label"] for r in rows]
        answered = [(s, g) for s, g in zip(said, truth) if s != "null"]
        clear_no = [s for s, g in zip(said, truth) if g in NO]
        out[f"t{t}"] = {
            "accuracy": round(sum(s == g for s, g in zip(said, truth)) / len(rows), 3),
            "side_accuracy": round(sum((s in YES) == (g in YES) for s, g in answered) / max(1, len(answered)), 3),
            "abstain": round(1 - len(answered) / len(rows), 3),
            # what poisons sleep: a clear no read as a yes
            "false_accept": round(sum(s in YES for s in clear_no) / max(1, len(clear_no)), 3),
            "missed_null": round(sum(s != "null" for s, g in zip(said, truth) if g == "null") / max(1, truth.count("null")), 3),
        }
        if t == 0.0:
            for s, g in zip(said, truth):
                out["confusion"].setdefault(g, {}).setdefault(s, 0)
                out["confusion"][g][s] += 1
    if "reward" in rows[0]:  # shopping: against the hidden true reward too
        said = [LABELS[p] for p in pred.tolist()]
        bad = [s for s, r in zip(said, rows) if r["reward"] < 0.5]
        out["accept_when_reward_below_0.5"] = round(sum(s in YES for s in bad) / max(1, len(bad)), 3)
    return out


@torch.no_grad()
def evaluate(model, scorer: Scorer, sets: dict, batch: int = 32) -> dict:
    model.eval()
    out = {}
    for name, rows in sets.items():
        probs = torch.cat([F.softmax(scorer.logits(model, rows[i:i + batch]), -1).cpu() for i in range(0, len(rows), batch)])
        out[name] = metrics(rows, probs)
    model.train()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("lab/data/sense"))
    p.add_argument("--out", type=Path, default=Path("lab/data/sense/adapter"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval-limit", type=int, default=2000, help="rows per evaluation set")
    args = p.parse_args()
    status = Status()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL)
    scorer = Scorer(tok, dev)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    train = load(args.data / "train.jsonl")
    sets = {name: load(args.data / f"{name}.jsonl", args.eval_limit) for name in ("val", "test_wildfb", "test_shopping")}
    status.update(stage="zero-shot evaluation", model=MODEL, lora={"rank": RANK, "alpha": ALPHA, "modules": MODULES},
                  train={"rows": len(train), "step": 0, "steps": 0, "loss": []}, force=True)

    zero = evaluate(base, scorer, sets)
    status.update(eval={"zero_shot": zero}, force=True)

    model = get_peft_model(base, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05, target_modules=MODULES,
                                            task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    steps = math.ceil(len(train) * args.epochs / args.batch)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / steps))
    status.update(stage="training", train={"steps": steps, "trainable_params": params}, force=True)

    rng, losses, t0 = random.Random(0), [], time.time()
    order = []
    model.train()
    for step in range(steps):
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        rows = [train[order.pop()] for _ in range(min(args.batch, len(order)))]
        target = torch.tensor([LABELS.index(r["label"]) for r in rows], device=dev)
        loss = F.cross_entropy(scorer.logits(model, rows), target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % 25 == 0 or step + 1 == steps:
            avg = sum(losses[-25:]) / len(losses[-25:])
            per_step = (time.time() - t0) / (step + 1)
            status.update(train={"step": step + 1, "loss": status.state["train"]["loss"] + [[step + 1, round(avg, 4)]],
                                 "eta_s": round(per_step * (steps - step - 1))})

    status.update(stage="evaluating", force=True)
    trained = evaluate(model, scorer, sets)
    model.save_pretrained(str(args.out))
    status.update(stage="done", eval={"trained": trained}, adapter=str(args.out), force=True)
    print(json.dumps({"zero_shot": zero, "trained": trained}, indent=2))


if __name__ == "__main__":
    main()

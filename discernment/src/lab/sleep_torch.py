"""Sleep on a GPU: the same LoRA as lab/sleep.py, trained with PyTorch + PEFT, plus preference learning.

Two lessons per night, mixed in every step:
  - imitation (SFT): the creature's replies in attempts judged good, prompt masked out;
  - preference (DPO): at the step where a corrected retry first differed from the failed attempt, prefer the
    corrected reply. The reference is the creature as it woke up this morning (before tonight's update).
The adapter is saved in PEFT format so vLLM can load it into the running server.

Ring brains: frozen rings (earlier LoRAs) are merged into the base before tonight's ring is trained, so only
the newest ring learns; --rank sets a new ring's size. --lr-scale shrinks the learning rate (lr-decay brain).

Run as a subprocess by lab/sleep.py:
  python -m lab.sleep_torch <night_dir> <model> [previous_adapter] [--frozen a,b] [--rank 8] [--lr-scale 1]
"""

import argparse

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.sleep import BATCH, EPOCHS, LEARNING_RATE, LORA_LAYERS, MAX_ITERS, MAX_SEQ

RANK, SCALE = 8, 20.0  # mlx-lm's defaults, so a GPU night trains the same LoRA as a Mac night
DPO_BETA = 0.1
MODULES = "(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def encode(tok, prompt: list[dict], reply: str) -> tuple[list[int], list[int]]:
    """Token ids and labels (-100 on the prompt), left-truncating the prompt to fit MAX_SEQ."""
    p = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(p, add_special_tokens=False).input_ids
    c_ids = tok(reply + "<|im_end|>", add_special_tokens=False).input_ids[: MAX_SEQ // 2]
    p_ids = p_ids[-(MAX_SEQ - len(c_ids)):]
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids


def encode_text(tok, prompt_text: str, reply_text: str) -> tuple[list[int], list[int]]:
    """As encode, for an example already rendered to text (E10: a τ²-bench turn with its tools, rendered by the
    chat template exactly as vLLM saw it); reply_text ends with <|im_end|>."""
    p_ids = tok(prompt_text, add_special_tokens=False).input_ids
    c_ids = tok(reply_text, add_special_tokens=False).input_ids[: MAX_SEQ // 2]
    p_ids = p_ids[-(MAX_SEQ - len(c_ids)):]
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids


def collate(rows, pad_id, dev):
    n = max(len(ids) for ids, _ in rows)
    ids = torch.tensor([i + [pad_id] * (n - len(i)) for i, _ in rows], device=dev)
    labels = torch.tensor([lab + [-100] * (n - len(lab)) for _, lab in rows], device=dev)
    mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in rows], device=dev)
    return ids, labels, mask


def logprobs(model, ids, labels, mask) -> torch.Tensor:
    """Summed log-probability of the labelled (reply) tokens, per row. Only reply positions go through the
    vocabulary projection, so memory stays small however long the prompt is."""
    base = model.get_base_model()
    hidden = base.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    target = labels[:, 1:]
    keep = target != -100
    logits = base.lm_head(hidden[keep]).float()
    lp = torch.gather(F.log_softmax(logits, -1), 1, target[keep].unsqueeze(-1)).squeeze(-1)
    rows = torch.zeros(ids.shape[0], device=ids.device).index_add(0, keep.nonzero()[:, 0], lp)
    return rows, keep.sum(-1)


def load(model_name: str, previous: str | None, dev: str, frozen: list[str] = (), rank: int = RANK):
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(dev)
    for ring in frozen:  # earlier rings become part of the fixed brain
        base = PeftModel.from_pretrained(base, ring).merge_and_unload()
    if previous:
        return PeftModel.from_pretrained(base, previous, is_trainable=True)
    n = base.config.num_hidden_layers
    layers = "|".join(str(i) for i in range(n - LORA_LAYERS, n))
    config = LoraConfig(r=rank, lora_alpha=rank * SCALE, lora_dropout=0.0,
                        target_modules=f".*layers\\.({layers})\\.{MODULES}", task_type="CAUSAL_LM")
    return get_peft_model(base, config)


def train(night_dir: Path, model_name: str, previous: str | None, frozen: list[str] = (), rank: int = RANK,
          lr_scale: float = 1.0) -> None:
    examples = [json.loads(line)["messages"] for line in (night_dir / "data" / "train.jsonl").open()]
    pairs_file = night_dir / "data" / "pairs.jsonl"
    pairs = [json.loads(line) for line in pairs_file.open()] if pairs_file.exists() else []
    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = load(model_name, previous, dev, frozen, rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    rng = random.Random(0)
    sft = [encode_text(tok, m[-1]["prompt_text"], m[-1]["content"]) if "prompt_text" in m[-1]
           else encode(tok, m[:-1], m[-1]["content"]) for m in examples]
    weight = [float(m[-1].get("weight", 1.0)) for m in examples]  # E5: an "acceptable" (80) lesson counts half
    dpo = [(encode(tok, p["prompt"], p["chosen"]), encode(tok, p["prompt"], p["rejected"])) for p in pairs]

    # Reference: the creature before tonight (LoRA B starts at zero, so night 1's reference is the base model).
    model.eval()
    ref = []
    with torch.no_grad():
        for chosen, rejected in dpo:
            lp, _ = logprobs(model, *collate([chosen, rejected], pad, dev))
            ref.append(lp.tolist())

    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE * lr_scale)
    iters = min(MAX_ITERS, max(10, (len(sft) * EPOCHS) // BATCH))
    log = (night_dir / "train.log").open("w")
    log.write(f"device {dev}; {len(sft)} examples, {len(dpo)} pairs, {iters} iters; {len(frozen)} frozen rings, "
              f"rank {rank}, lr x{lr_scale:.3f}\n")
    for it in range(1, iters + 1):
        loss, parts = 0.0, {}
        if sft:
            picks = rng.sample(range(len(sft)), min(BATCH, len(sft)))
            lp, n = logprobs(model, *collate([sft[i] for i in picks], pad, dev))
            w = torch.tensor([weight[i] for i in picks], device=dev)
            parts["sft"] = -((w * lp).sum() / (w * n).sum())
        if dpo:
            picks = rng.sample(range(len(dpo)), min(BATCH, len(dpo)))
            rows = [r for i in picks for r in dpo[i]]  # chosen, rejected, chosen, rejected, ...
            lp, _ = logprobs(model, *collate(rows, pad, dev))
            ref_lp = torch.tensor([v for i in picks for v in ref[i]], device=dev)
            margin = (lp[0::2] - ref_lp[0::2]) - (lp[1::2] - ref_lp[1::2])
            parts["dpo"] = -F.logsigmoid(DPO_BETA * margin).mean()
        loss = sum(parts.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 10 == 0 or it == iters:
            log.write(f"iter {it}: " + ", ".join(f"{k} {v.item():.4f}" for k, v in parts.items()) + "\n")
            log.flush()
    model.save_pretrained(str(night_dir / "adapter"))
    log.close()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("night_dir", type=Path)
    a.add_argument("model")
    a.add_argument("previous", nargs="?")
    a.add_argument("--frozen", default="", help="comma-separated adapters merged into the base, oldest first")
    a.add_argument("--rank", type=int, default=RANK)
    a.add_argument("--lr-scale", type=float, default=1.0)
    args = a.parse_args()
    train(args.night_dir, args.model, args.previous, [f for f in args.frozen.split(",") if f], args.rank, args.lr_scale)

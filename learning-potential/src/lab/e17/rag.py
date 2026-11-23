"""E17: retrieved experience on the job (Synapse-style). A frozen tiny model does E15's practice tasks one after another;
every finished task is logged verbatim (the user's requests and the calls, turn by turn), and before each task the
most similar logged tasks are retrieved and shown as worked examples. The weights never change; nothing is summarised.

Arms:
  teach  oracle teaching: each task is logged with its expected calls
  judge  oracle judging: each task the model passed is logged with the model's own calls

Retrieval: a small embedding model (all-MiniLM-L6-v2, on the CPU) embeds the user's requests; among logged tasks that
use the same tools (the same set of tool classes), the closest in meaning come first; up to --k of them within --budget
tokens go into one system message at the start of the task, under a short neutral header, the way E15's notes go in.
Calls are shown in BFCL's reference form for every model; each model still answers in its own format.

    python -m lab.e17.rag --registry Qwen/Qwen3-1.7B-FC --data <v3> <v4> <v5> --out <dir> --arm teach \
        [--model-path <local weights>] [--tasks 1200] [--seed 15] [--k 3] [--budget 1200]

Writes <out>/log.jsonl, <out>/memory.jsonl (the log, one task a line), <out>/memory-<n>.jsonl every 100 tasks (for the
probes) and <out>/status.json; resumes from <out>/state.json after a crash.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from lab.e15.notes import SPLIT, Runner, load_pool

HEADER = "Earlier tasks with these tools and the calls that completed them:"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """The small embedding model, on the CPU (it is about 22M parameters; the GPU stays with vLLM)."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(EMBEDDER)
        self.model = AutoModel.from_pretrained(EMBEDDER).eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            b = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = self.torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
        return v.numpy()


def requests(q: dict) -> list[str]:
    return [" ".join(m["content"] for m in turn if m["role"] == "user") for turn in q["question"]]


def experience(q: dict, calls: list[list[str]]) -> dict:
    """One logged task: its tools, its user requests and the calls, turn by turn."""
    return {"tools": sorted(q["involved_classes"]), "requests": requests(q), "calls": calls}


def show(e: dict) -> str:
    lines = []
    for user, calls in zip(e["requests"], e["calls"]):
        lines.append(f"  User: {user}")
        lines.append("  Calls: " + ("; ".join(calls) if calls else "(none)"))
    return "\n".join(lines)


class Memory:
    def __init__(self, embed: Embedder):
        self.embed, self.items, self.vecs = embed, [], np.zeros((0, 384), dtype=np.float32)

    def add(self, e: dict):
        self.items.append(e)
        self.vecs = np.vstack([self.vecs, self.embed([" \n".join(e["requests"])])])

    def load(self, path: Path, n: int | None = None):
        items = []
        for line in open(path):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off by a lost VM
                break
        items = items[:n] if n is not None else items
        self.items = items
        self.vecs = self.embed([" \n".join(e["requests"]) for e in items]) if items else np.zeros((0, 384), np.float32)

    def retrieve(self, q: dict, k: int, budget: int, ntok) -> tuple[str, list[int]]:
        """The system message for task q (empty when nothing fits) and the logged tasks it shows."""
        if not self.items:
            return "", []
        tools = sorted(q["involved_classes"])
        same = [i for i, e in enumerate(self.items) if e["tools"] == tools]
        if not same:
            return "", []
        v = self.embed([" \n".join(requests(q))])[0]
        order = sorted(same, key=lambda i: -float(self.vecs[i] @ v))
        chosen, parts, used = [], [], ntok(HEADER)
        for i in order:
            text = f"Example {len(chosen) + 1}\n" + show(self.items[i])
            t = ntok(text)
            if used + t > budget:
                continue  # too long for what is left; a shorter one may still fit
            chosen.append(i)
            parts.append(text)
            used += t
            if len(chosen) == k:
                break
        return (HEADER + "\n\n" + "\n\n".join(parts) if parts else ""), chosen


class RagRunner(Runner):
    def solve_with(self, i: int, q: dict, gt: list, system: str) -> dict:
        """E15's solve with the retrieved examples in place of the notes."""
        q = copy.deepcopy(q)
        if system:
            q["question"][0].insert(0, {"role": "system", "content": system})
        return self.solve(i, q, gt, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["teach", "judge"], required=True)
    ap.add_argument("--tasks", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--k", type=int, default=3, help="examples shown at most")
    ap.add_argument("--budget", type=int, default=1200, help="tokens for the examples and their header")
    a = ap.parse_args()
    a.temp = 0.7

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(a.data, a.seed, Path(a.split) if a.split else None)[: a.tasks]
    run = RagRunner(a)
    mem = Memory(Embedder())

    st = {"done": 0, "logged": 0, "started": time.time()}
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        print(f"resumed after task {st['done']}", flush=True)
    log = []
    if (out / "log.jsonl").exists():
        for line in open(out / "log.jsonl"):
            try:
                log.append(json.loads(line))
            except json.JSONDecodeError:
                break
    log = log[: st["done"]]
    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    if (out / "memory.jsonl").exists():
        mem.load(out / "memory.jsonl", st["logged"])
    (out / "memory.jsonl").write_text("".join(json.dumps(e) + "\n" for e in mem.items))

    def status(stage):
        ok = [r["valid"] for r in log]
        shown = [r["shown"] for r in log]
        (out / "status.json").write_text(json.dumps({
            "arm": a.arm, "registry": a.registry, "seed": a.seed, "stage": stage, "done": len(log), "tasks": len(pool),
            "success": round(100 * sum(ok) / len(ok), 1) if ok else None,
            "last50": round(100 * sum(ok[-50:]) / len(ok[-50:]), 1) if ok else None,
            "memory": len(mem.items), "examples_per_task": round(sum(shown) / len(shown), 2) if shown else None,
            "secs_per_task": round(sum(r["secs"] for r in log) / len(log), 1) if log else None,
            "started": st["started"], "updated": time.time(),
            "settings": {k: v for k, v in vars(a).items() if k not in ("data", "out")}}))

    status("running")
    for i in range(st["done"], len(pool)):
        src, q, gt = pool[i]
        system, chosen = mem.retrieve(q, a.k, a.budget, run.ntok)
        s = run.solve_with(i, q, gt, system)
        if a.arm == "teach":
            e = experience(q, [list(t) for t in gt])
        else:
            e = experience(q, [[c for c in t if not c.startswith("(text")] for t in run.decoded(s["result"])]) \
                if s["valid"] else None
        if e:
            mem.add(e)
            with open(out / "memory.jsonl", "a") as f:
                f.write(json.dumps(e) + "\n")
        r = {"i": i, "set": src, "id": q["id"], "valid": s["valid"], "secs": round(s["secs"], 1),
             "error": (s["error"] or {}).get("error_type"), "shown": len(chosen), "memory": len(mem.items)}
        log.append(r)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        st["done"], st["logged"] = len(log), len(mem.items)
        if st["done"] % 100 == 0:  # a snapshot for the probes
            (out / f"memory-{st['done']}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in mem.items))
        tmp = out / "state.tmp"
        tmp.write_text(json.dumps(st))
        tmp.replace(out / "state.json")
        status("running")
    status("done")


if __name__ == "__main__":
    main()

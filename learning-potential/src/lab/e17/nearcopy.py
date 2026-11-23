"""E17 check before any run: are held-out (and probe) tasks near-copies of practice tasks? Retrieval could then copy an
almost identical answer and "unseen" scores would overstate learning. For each held-out task, against the 1,200
practice tasks of data/split.json:

  - meaning: cosine similarity of the user's requests (all turns), with the small embedding model E17 retrieves with,
    to the closest practice task;
  - same answer: a practice task whose expected calls are exactly the same, turn by turn (a retrieved copy would be the
    answer);
  - same functions: a practice task whose expected calls use the same functions in the same order, turn by turn
    (arguments may differ).

    python -m lab.e17.nearcopy --data <v3> <v4> <v5> [--out nearcopy.json]
"""
import argparse
import ast
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

SPLIT = Path(__file__).resolve().parents[3] / "data" / "split.json"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def load(dirs):
    out = {}
    for d in dirs:
        d = Path(d)
        gts = {json.loads(l)["id"]: json.loads(l)["ground_truth"]
               for l in open(d / "possible_answer/BFCL_v4_multi_turn_base.json")}
        for l in open(d / "BFCL_v4_multi_turn_base.json"):
            q = json.loads(l)
            out[(d.name, q["id"])] = (q, gts[q["id"]])
    return out


def text(q) -> str:
    return " \n".join(m["content"] for turn in q["question"] for m in turn if m["role"] == "user")


def functions(gt) -> tuple:
    def name(c):
        try:
            return ast.unparse(ast.parse(c.strip(), mode="eval").body.func)
        except SyntaxError:
            return c.split("(")[0]
    return tuple(tuple(name(c) for c in turn) for turn in gt)


@torch.no_grad()
def embed(texts, bs=64):
    tok = AutoTokenizer.from_pretrained(EMBEDDER)
    model = AutoModel.from_pretrained(EMBEDDER).eval()
    vecs = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i: i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)  # mean pooling, as the model is meant to be used
        vecs.append(torch.nn.functional.normalize(v, dim=-1))
    return torch.cat(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="nearcopy.json")
    a = ap.parse_args()
    sp = json.loads(SPLIT.read_text())
    tasks = load(a.data)
    practice = [tuple(k) for k in sp["practice"]]
    probe = {tuple(k) for k in sp["probe"]}
    heldout = [tuple(k) for k in sp["heldout"]]

    P = embed([text(tasks[k][0]) for k in practice])
    H = embed([text(tasks[k][0]) for k in heldout])
    sim = H @ P.T
    best, idx = sim.max(dim=1)

    p_ans = {}
    p_fun = {}
    for k in practice:
        gt = tasks[k][1]
        p_ans.setdefault(json.dumps(gt), []).append(k)
        p_fun.setdefault(functions(gt), []).append(k)

    rows = []
    for i, k in enumerate(heldout):
        q, gt = tasks[k]
        near = practice[int(idx[i])]
        rows.append({"set": k[0], "id": k[1], "probe": k in probe, "cos": round(float(best[i]), 4),
                     "nearest": list(near), "nearest_same_answer": tasks[near][1] == gt,
                     "nearest_same_functions": functions(tasks[near][1]) == functions(gt),
                     "any_same_answer": json.dumps(gt) in p_ans,
                     "any_same_functions": functions(gt) in p_fun,
                     "text": text(q)[:200], "nearest_text": text(tasks[near][0])[:200]})
    Path(a.out).write_text(json.dumps(rows, indent=1))

    def report(name, rs):
        n = len(rs)
        cos = sorted(r["cos"] for r in rs)
        pct = lambda f: f"{sum(map(f, rs))} ({100 * sum(map(f, rs)) / n:.0f}%)"
        print(f"{name}: {n} tasks")
        print(f"  closest practice task in meaning: median cos {cos[n // 2]:.3f}, 90th pct {cos[int(n * .9)]:.3f}, max {cos[-1]:.3f}")
        for t in (0.95, 0.9, 0.8):
            print(f"  cos >= {t}: {pct(lambda r, t=t: r['cos'] >= t)}")
        print(f"  some practice task has exactly the same expected calls: {pct(lambda r: r['any_same_answer'])}")
        print(f"  ... and it is the one retrieval would find first: {pct(lambda r: r['nearest_same_answer'])}")
        print(f"  some practice task uses the same functions in the same order: {pct(lambda r: r['any_same_functions'])}")
        print(f"  ... and it is the closest one: {pct(lambda r: r['nearest_same_functions'])}")
    report("probe", [r for r in rows if r["probe"]])
    report("held-out (all)", rows)


if __name__ == "__main__":
    main()

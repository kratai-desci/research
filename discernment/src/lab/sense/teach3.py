"""The v3 teacher: Qwen3-32B reads each reaction under guide3.md, says why in one line, then answers v3's two
questions: is it a clear verdict on the result (yes / no), and which score (0, 20, 80, 100).

The teacher answers "why: ...\\nclear: yes|no\\nscore: <score>"; the likelihoods come from the log-probabilities
of the first token after "clear:" and after "score:", so the student learns the teacher's uncertainty, not only
its pick. It always gives a score, even when not clear; the student learns the score in proportion to clarity.

  uv run python -m lab.sense.teach3 --select lab/data/sense2/scored.jsonl --gaps lab/data/sense3/gaps.jsonl \\
      --raw lab/data/sense3/raw.jsonl                                    # the rows to score (a balanced sample)
  uv run python -m lab.sense.teach3 --raw lab/data/sense3/raw.jsonl --out lab/data/sense3/scored.jsonl
"""

import argparse
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lab.sense import CLEAR, NEXT_MOVES, SCORES
from lab.sense.teach import TASKS, TEACHER, user_text

GUIDE = (Path(__file__).parent / "guide3.md").read_text()
SYSTEM = (GUIDE + "\n\n## Your answer\n\nAnswer in exactly five lines and nothing else:\n"
          "why: <one short sentence: what the words say about the result>\nclear: <yes or no>\n"
          "score: <100, 80, 20 or 0: the score if it is clear; if it is not clear, your best guess anyway>\n"
          f"move: <one of: {', '.join(NEXT_MOVES)}>\ntask: <one of: {', '.join(TASKS)}>")
PER_CELL = 2000  # rows sampled per (task type, v2 teacher's best label) from the v2 data


def _after(choice: dict, key: str) -> dict[str, float] | None:
    """The top log-probabilities of the first non-space token after "<key>:" in the generated text."""
    tokens, tops, text = choice["logprobs"]["tokens"], choice["logprobs"]["top_logprobs"], ""
    for i, t in enumerate(tokens):
        if text.rstrip().endswith(f"{key}:") and t.strip():
            return tops[i]
        text += t
    return None


def _dist(top: dict[str, float] | None, labels: list[str]) -> list[float] | None:
    """Probabilities over labels, matching each top token by its first character (" 100" and "1" both mean 100)."""
    if not top:
        return None
    first = {lab[0]: lab for lab in labels}
    mass = defaultdict(float)
    for tok, lp in top.items():
        lab = first.get(tok.strip()[:1].lower())
        if lab:
            mass[lab] += math.exp(lp)
    total = sum(mass.values())
    return [round(mass[lab] / total, 4) for lab in labels] if total else None


class Teacher3:
    def __init__(self, base_url: str):
        import requests
        from transformers import AutoTokenizer

        self.http, self.url = requests.Session(), base_url.rstrip("/")
        self.tok = AutoTokenizer.from_pretrained(TEACHER)

    def score(self, row: dict) -> dict:
        text = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(row)}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        r = self.http.post(f"{self.url}/v1/completions", timeout=300, json={
            "model": TEACHER, "prompt": text, "max_tokens": 120, "temperature": 0, "logprobs": 20})
        r.raise_for_status()
        choice = r.json()["choices"][0]
        clear, score = _dist(_after(choice, "clear"), CLEAR), _dist(_after(choice, "score"), SCORES)
        lines = {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in choice["text"].splitlines())}
        move = lines.get("move") if lines.get("move") in NEXT_MOVES else None
        task = lines.get("task") if lines.get("task") in TASKS else None
        out = {**row, "why": lines.get("why", ""), "next_move": row.get("next_move") or move, "task": row.get("task") or task}
        if clear and score:
            out["teacher3"] = {"clear": clear[0], "score": dict(zip(SCORES, score))}
        return out


def label(t: dict) -> str:
    """The teacher's single best answer: null when not clear, else its most likely score."""
    return "null" if t["clear"] < 0.5 else max(t["score"], key=t["score"].get)


def select(scored: Path, gaps: list[Path], out: Path) -> None:
    """A balanced sample of the v2 rows (by task and v2 label) plus every gap reaction, unscored."""
    rng, cells = random.Random(0), defaultdict(list)
    for line in scored.open():
        r = json.loads(line)
        if r.get("teacher"):
            cells[(r.get("task") or "other", max(r["teacher"], key=r["teacher"].get))].append(r)
    keep = [r for cell in cells.values() for r in rng.sample(cell, min(len(cell), PER_CELL))]
    for g in gaps:
        keep += [json.loads(line) for line in g.open()] if g.exists() else []
    fields = ("source", "task", "request", "assistant", "said", "next_move", "meta")
    with out.open("w") as f:
        for r in keep:
            f.write(json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False) + "\n")
    print(f"selected {len(keep)} rows into {out}")


def check(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open() if "teacher3" in line]
    rows = [r for r in rows if r.get("teacher3")]
    out = {"rows": len(rows), "labels": {lab: sum(label(r["teacher3"]) == lab for r in rows) for lab in SCORES + ["null"]}}
    gap = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["source"] == "generated":
            gap[r["meta"]["gap"]][label(r["teacher3"])] += 1
    out["gaps"] = {g: dict(v) for g, v in gap.items()}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--select", type=Path, help="v2 scored rows to sample from (writes --raw and stops)")
    p.add_argument("--gaps", type=Path, nargs="*", default=[])
    p.add_argument("--raw", type=Path, default=Path("lab/data/sense3/raw.jsonl"))
    p.add_argument("--out", type=Path, default=Path("lab/data/sense3/scored.jsonl"))
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--limit", type=int, default=0, help="score only N random rows (a check run)")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--check", type=Path)
    args = p.parse_args()
    if args.check:
        print(json.dumps(check(args.check), indent=2))
        return
    if args.select:
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        select(args.select, args.gaps, args.raw)
        return
    from lab.sense.status import Status

    status = Status(args.out.parent / "status.json", remote="sense3/status.json")
    rows = [json.loads(line) for line in args.raw.open()]
    if args.limit:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    done = {json.loads(line)["_i"] for line in args.out.open()} if args.out.exists() else set()  # resume
    teacher = Teacher3(args.url)
    status.update(stage="teacher scoring", teach={"rows": len(rows), "done": len(done)}, force=True)
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]
    with args.out.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        for n, out in enumerate(pool.map(lambda ir: {**teacher.score(ir[1]), "_i": ir[0]}, todo), 1):
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                f.flush()
                status.update(teach={"done": len(done) + n})
    status.update(stage="teacher done", teach={"done": len(rows)}, check=check(args.out), force=True)


if __name__ == "__main__":
    main()

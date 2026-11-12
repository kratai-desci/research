"""The teacher: Qwen3-32B reads each reaction under the labelling guide and gives the five scores' likelihoods.

One call per row, served by vLLM. The teacher answers "<score>\\nmove: <next move>\\ntask: <task type>"; the
likelihoods of the five scores come from the first answer token's log-probabilities (the teacher shares the
student's tokenizer, so the label tokens are the same), and the next move and task type from the text.
The teacher also sees the assistant's delivery as context; the student never will.

  uv run python -m lab.sense.teach --raw lab/data/sense2/raw.jsonl --out lab/data/sense2/scored.jsonl
  uv run python -m lab.sense.teach --check lab/data/sense2/scored.jsonl     # agreement with the datasets' own labels
"""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lab.sense import LABELS, NEXT_MOVES

TEACHER = "Qwen/Qwen3-32B-AWQ"
TASKS = ["coding", "writing", "maths and science", "questions and facts", "advice", "creative and roleplay",
         "translation", "booking and ordering", "recommending", "support and troubleshooting", "other"]
GUIDE = (Path(__file__).parent / "guide.md").read_text()
SYSTEM = (GUIDE + "\n\n## Your answer\n\nAnswer in exactly three lines and nothing else:\n<score: 100, 80, 20, 0 or null>\n"
          f"move: <one of: {', '.join(NEXT_MOVES)}>\ntask: <one of: {', '.join(TASKS)}>")


def user_text(row: dict) -> str:
    move = row.get("next_move") or "(not given: infer it)"
    return (f"Request: {row['request'] or '(not shown)'}\nWhat the assistant did (context only; score the person, not the "
            f"work): {row.get('assistant') or '(not shown)'}\nNext move: {move}\nWords: {row['said'] or '(nothing)'}")


class Teacher:
    def __init__(self, base_url: str):
        import requests
        from transformers import AutoTokenizer

        self.http, self.url = requests.Session(), base_url.rstrip("/")
        self.tok = AutoTokenizer.from_pretrained(TEACHER)
        self.label_tokens = [self.tok.decode([self.tok(lab, add_special_tokens=False).input_ids[0]]) for lab in LABELS]

    def score(self, row: dict) -> dict:
        text = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(row)}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        r = self.http.post(f"{self.url}/v1/completions", timeout=300, json={
            "model": TEACHER, "prompt": text, "max_tokens": 24, "temperature": 0, "logprobs": 20})
        r.raise_for_status()
        choice = r.json()["choices"][0]
        top = choice["logprobs"]["top_logprobs"][0]
        logits = [top.get(t, -math.inf) for t in self.label_tokens]
        m = max(logits)
        probs = [0.0] * 4 + [1.0] if m == -math.inf else [math.exp(x - m) for x in logits]
        probs = [p / sum(probs) for p in probs]
        lines = {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in choice["text"].splitlines()[1:])}
        move = lines.get("move") if lines.get("move") in NEXT_MOVES else None
        task = lines.get("task") if lines.get("task") in TASKS else None
        return {**row, "teacher": dict(zip(LABELS, (round(p, 4) for p in probs))),
                "next_move": row.get("next_move") or move, "task": row.get("task") or task}


def check(path: Path) -> dict:
    """How the teacher's best guess lines up with labels the datasets carry themselves."""
    rows = [json.loads(line) for line in path.open()]
    best = lambda r: max(r["teacher"], key=r["teacher"].get)
    side = lambda lab: "yes" if lab in ("80", "100") else "no" if lab in ("0", "20") else "null"
    out = {"rows": len(rows), "teacher_labels": {lab: sum(best(r) == lab for r in rows) for lab in LABELS}}
    fb = [r for r in rows if r["source"] == "wildfb"]
    if fb:  # WildFB's 1-4: 1 rejection, 2 correction, 3 positive engagement, 4 praise
        want = {1: "no", 2: "no", 3: "yes", 4: "yes"}
        hit = [side(best(r)) == want[r["meta"]["wildfb_label"]] for r in fb if side(best(r)) != "null"]
        out["wildfb_side_agreement"] = round(sum(hit) / max(1, len(hit)), 3)
        out["wildfb_teacher_unclear"] = round(sum(side(best(r)) == "null" for r in fb) / len(fb), 3)
    wf = [r for r in rows if r["source"] == "wildfeedback" and (r["meta"].get("sat") or r["meta"].get("dsat"))]
    if wf:  # WildFeedback's satisfaction / dissatisfaction flags
        hit = [side(best(r)) == ("yes" if r["meta"]["sat"] else "no") for r in wf if side(best(r)) != "null"]
        out["wildfeedback_side_agreement"] = round(sum(hit) / max(1, len(hit)), 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("lab/data/sense2/raw.jsonl"))
    p.add_argument("--out", type=Path, default=Path("lab/data/sense2/scored.jsonl"))
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--limit", type=int, default=0, help="score only the first N rows (a check run)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--check", type=Path)
    args = p.parse_args()
    if args.check:
        print(json.dumps(check(args.check), indent=2))
        return
    from lab.sense.status import Status

    status = Status(args.out.parent / "status.json", remote="sense2/status.json")
    rows = [json.loads(line) for line in args.raw.open()]
    if args.limit:
        import random

        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    done = {json.loads(line)["_i"] for line in args.out.open()} if args.out.exists() else set()  # resume
    teacher = Teacher(args.url)
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

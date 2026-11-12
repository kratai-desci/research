"""A test of the sense on reactions from our own runs: the WebShop v2 user's words (Gemini) with the behaviour it
was told to show, which a yes must match (praise and accept are the only behaviours that should read as a yes).

The sense never trains on these; they are only a check before a sense goes into a life. Up to --per-kind
reactions per behaviour, from the life diaries (lives/<life>/diary.jsonl) of the experiments given.

  uv run python -m lab.sense.live --diaries lab/data/live-diaries --out lab/data/sense3/live.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--diaries", type=Path, required=True, help="a folder of <life>/diary.jsonl")
    p.add_argument("--out", type=Path, default=Path("lab/data/sense3/live.jsonl"))
    p.add_argument("--per-kind", type=int, default=300)
    args = p.parse_args()
    by = defaultdict(list)
    for diary in sorted(args.diaries.glob("*/diary.jsonl")):
        for line in diary.open():
            r = json.loads(line)
            if r.get("persona") in (None, "", "fallback") or "reaction" not in r:
                continue  # template-user lives, or a plain fallback sentence
            by[r["reaction"]].append({"request": r["instruction"].removeprefix("Instruction: "), "next_move": r["next_move"],
                                      "said": r["said"], "kind": r["reaction"], "persona": r["persona"],
                                      "reward": r["reward_hidden"], "life": diary.parent.name})
    rng = random.Random(0)
    rows = [x for kind in sorted(by) for x in rng.sample(by[kind], min(len(by[kind]), args.per_kind))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} reactions: " + ", ".join(f"{k} {min(len(v), args.per_kind)}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    main()

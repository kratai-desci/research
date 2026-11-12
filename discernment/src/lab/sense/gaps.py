"""Reactions the real datasets rarely have, written by the teacher (Qwen3-32B) and then scored by it like any row.

Gaps: politely unhappy, settling, vague, sarcastic, "not my style", strong rejection, building on a result, plain
acceptance. Task types exclude shopping (the unseen test). The personas here differ from the E6 simulated user's
(terse, chatty, picky, polite, easygoing, blunt), and the E6 user is worded by Gemini, so the sense never trains
on its own test user's style.

  uv run python -m lab.sense.gaps --out lab/data/sense2/gaps.jsonl
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lab.sense.teach import TEACHER, TASKS, Teacher

GAPS = {
    "polite_unhappy": "thanks the assistant politely but is clearly not satisfied and asks for it again or hints it missed",
    "settling": "reluctantly settles for the result without saying whether it is good ('I guess it'll do', 'whatever')",
    "vague": "replies vaguely, so you cannot tell whether it worked ('hmm', 'ok...', 'interesting')",
    "sarcastic": "is sarcastic about the result (praise that is really a complaint)",
    "not_my_style": "rejects the result as not to their taste ('not my style', 'not really my thing')",
    "strong_reject": "rejects the result firmly because it missed what they asked for",
    "builds_on": "is happy enough and builds on the result with a new request or a new preference they had not mentioned",
    "accepts": "accepts the result simply and moves on, without praise",
    # v3: the gap E6 found. Warm words that thank the effort, not the result, after it did not work
    "thanks_for_effort": "did not get what they needed, and warmly thanks the assistant for trying anyway (not for the "
                         "result), often saying they will look elsewhere or leave it for now",
}
PERSONAS = ["a busy professional", "a teenager", "a non-native English speaker", "an older person new to AI",
            "someone who writes with lots of emoji", "a formal academic"]
TASK_TYPES = [t for t in TASKS if t != "other"]


def ask(teacher: Teacher, gap: str, task: str, persona: str, n: int, seed: int) -> list[dict]:
    prompt = (f"Write {n} different, realistic moments from conversations between a person and an AI assistant, about "
              f"{task}. In each, the assistant has just done something for the person, and the person {GAPS[gap]}. "
              f"The person is {persona}; vary the wording a lot. Reply with JSON only: a list of {n} objects with keys "
              '"request" (what the person asked), "assistant" (one sentence on what the assistant did), "said" (the '
              "person's exact next message).")
    text = teacher.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    r = teacher.http.post(f"{teacher.url}/v1/completions", timeout=600, json={
        "model": TEACHER, "prompt": text, "max_tokens": 2000, "temperature": 0.9, "seed": seed})
    r.raise_for_status()
    raw = r.json()["choices"][0]["text"].strip().removeprefix("```json").removesuffix("```").strip()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [{"source": "generated", "task": task, "request": str(i.get("request", ""))[:500],
             "assistant": str(i.get("assistant", ""))[:500], "said": str(i.get("said", ""))[:500], "next_move": None,
             "meta": {"gap": gap, "persona": persona}} for i in items if isinstance(i, dict) and i.get("said")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("lab/data/sense2/gaps.jsonl"))
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--rounds", type=int, default=3, help="passes over every gap x task x persona, 10 reactions each")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--only", nargs="*", default=[], help="write only these gap types (v3: thanks_for_effort)")
    p.add_argument("--no-score", action="store_true", help="write the reactions unscored (v3's teacher scores them)")
    args = p.parse_args()
    teacher = Teacher(args.url)
    gaps = args.only or list(GAPS)
    jobs = [(g, t, pe, rnd) for rnd in range(args.rounds) for g in gaps for t in TASK_TYPES for pe in PERSONAS]
    random.Random(0).shuffle(jobs)
    written = 0
    with args.out.open("w") as f, ThreadPoolExecutor(args.workers) as writers, ThreadPoolExecutor(args.workers) as scorers:
        made = writers.map(lambda ij: ask(teacher, *ij[1][:3], 10, ij[0]), enumerate(jobs))
        # Each written reaction is then scored by the teacher like a real one, for its soft labels.
        for rows in made:
            for row in (rows if args.no_score else scorers.map(teacher.score, rows)):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
    print(f"wrote {written} gap reactions from {len(jobs)} prompts")


if __name__ == "__main__":
    main()

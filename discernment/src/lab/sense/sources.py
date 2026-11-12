"""Sense v2's raw material: reactions from seven datasets we may use commercially, in one format.

Each row is one moment where an assistant delivered something and the user's next turn shows how they took it:
  {"source", "task", "request", "assistant", "said", "next_move", "meta"}
`request` is the user's last request before the delivery, `assistant` the delivery (context for the teacher only;
the sense never sees it), `said` the user's next turn ("" when the conversation ended there). `next_move` is set
when the data says it (the conversation ended, WildFeedback's turn state); otherwise the teacher decides it.

Assistant turns that only ask a question ("which town?") are skipped: the user's answer is not a reaction.
Shopping is never included; it is the unseen test domain.

Licences: WildFB MIT; WildFeedback, WildChat ODC-BY; MultiWOZ 2.2 MIT (source repo; Apache 2.0 on its HF card);
ReDial, CCPE-M, Taskmaster 1-3 CC BY 4.0. Attribution is owed for the ODC-BY and CC BY sets.

  uv run python -m lab.sense.sources --out lab/data/sense2/raw.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import requests

CAPS = {"wildfb": 30000, "wildfeedback": 30000, "multiwoz": 15000, "taskmaster": 20000, "redial": 15000, "ccpe": 5000}
CHARS = 500
GH = "https://raw.githubusercontent.com"


def cut(text: str, n: int = CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def english(text: str) -> bool:
    return bool(text) and sum(c.isascii() for c in text) / len(text) > 0.9


def delivers(text: str) -> bool:
    """An assistant turn that hands over something, rather than only asking."""
    t = text.strip()
    return len(t) > 15 and not t.endswith("?")


def get_json(url: str):
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.json()


def from_turns(source: str, task: str, turns: list[tuple[str, str]], meta: dict):
    """(speaker, text) turns with speakers 'user' / 'assistant' -> reaction rows."""
    for i, (who, text) in enumerate(turns):
        if who != "assistant" or not delivers(text):
            continue
        request = next((t for w, t in reversed(turns[:i]) if w == "user"), "")
        after = next(((w, t) for w, t in turns[i + 1:] if w == "user" or w == "assistant"), None)
        if not request:
            continue
        if after is None:
            yield {"source": source, "task": task, "request": cut(request), "assistant": cut(text), "said": "",
                   "next_move": "left the session", "meta": meta}
        elif after[0] == "user":
            yield {"source": source, "task": task, "request": cut(request), "assistant": cut(text), "said": cut(after[1]),
                   "next_move": None, "meta": meta}


def wildfb():
    from huggingface_hub import hf_hub_download

    for split in ("train", "test"):
        for line in open(hf_hub_download("THU-KEG/WildFB", f"{split}.jsonl", repo_type="dataset")):
            r = json.loads(line)
            users = [m["content"] for m in r["messages"] if m["role"] == "user"]
            bots = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
            fb = r["user_feedback"] if isinstance(r["user_feedback"], dict) else json.loads(r["user_feedback"])
            if users and bots and english(users[-1] + fb["content"]):
                yield {"source": "wildfb", "task": None, "request": cut(users[-1]), "assistant": cut(bots[-1]),
                       "said": cut(fb["content"]), "next_move": None, "meta": {"wildfb_label": r["label"], "split": split}}


WILDFEEDBACK_MOVE = {"NEWTOPIC": "new request", "REFINEMENT": "asked to adjust"}


def wildfeedback():
    """Utterance rows in order; a conversation restarts when TurnId falls back to 1 on a user row."""
    import ijson
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("microsoft/WildFeedback", "sat_dsat_annotation.json", repo_type="dataset")
    conv, last_turn = [], 0
    with open(path, "rb") as f:
        for row in ijson.items(f, "item"):
            if row.get("Role") == "User" and row.get("TurnId", 0) <= last_turn and conv:
                yield from _wildfeedback_conv(conv)
                conv = []
            last_turn = row.get("TurnId", 0)
            conv.append(row)
    if conv:
        yield from _wildfeedback_conv(conv)


def _wildfeedback_conv(conv):
    for i, row in enumerate(conv):
        if row.get("Role") != "User" or i < 2 or conv[i - 1].get("Role") != "Agent":
            continue
        request = next((r["Content"] for r in reversed(conv[:i - 1]) if r.get("Role") == "User"), "")
        said, bot = row.get("Content") or "", conv[i - 1].get("Content") or ""
        if not request or not english(request + said):
            continue
        yield {"source": "wildfeedback", "task": None, "request": cut(request), "assistant": cut(bot), "said": cut(said),
               "next_move": WILDFEEDBACK_MOVE.get(row.get("State")),
               "meta": {"state": row.get("State"), "sat": row.get("Satisfaction"), "dsat": row.get("Disatisfaction"),
                        "domain": row.get("Domain")}}


def multiwoz():
    for split, n in (("train", 17), ("dev", 2), ("test", 2)):
        for i in range(1, n + 1):
            for d in get_json(f"https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2/{split}/dialogues_{i:03d}.json"):
                turns = [("user" if t["speaker"] == "USER" else "assistant", t["utterance"]) for t in d["turns"]]
                yield from from_turns("multiwoz", "booking and ordering", turns, {"dialogue": d["dialogue_id"]})


def _utterances(source, task, convs, id_key):
    for c in convs:
        turns = [("user" if u["speaker"] == "USER" else "assistant", u["text"]) for u in c["utterances"]]
        yield from from_turns(source, task, turns, {"dialogue": c.get(id_key)})


TM2 = ["flights", "food-ordering", "hotels", "movies", "music", "restaurant-search", "sports"]


def taskmaster():
    for name in ("woz-dialogs.json", "self-dialogs.json"):
        yield from _utterances("taskmaster", "booking and ordering", get_json(f"{GH}/google-research-datasets/Taskmaster/master/TM-1-2019/{name}"), "conversation_id")
    for domain in TM2:
        task = "recommending" if domain in ("movies", "music") else "booking and ordering"
        yield from _utterances("taskmaster", task, get_json(f"{GH}/google-research-datasets/Taskmaster/master/TM-2-2020/data/{domain}.json"), "conversation_id")
    for i in range(20):
        yield from _utterances("taskmaster", "booking and ordering", get_json(f"{GH}/google-research-datasets/Taskmaster/master/TM-3-2020/data/data_{i:02d}.json"), "conversation_id")


def ccpe():
    yield from _utterances("ccpe", "recommending", get_json(f"{GH}/google-research-datasets/ccpe/master/data.json"), "conversationId")


def redial():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for split in ("train", "test"):
        table = pq.read_table(hf_hub_download("community-datasets/re_dial", f"data/{split}-00000-of-00001.parquet", repo_type="dataset"))
        for c in table.to_pylist():
            seeker = c["initiatorWorkerId"]
            turns = [("user" if m["senderWorkerId"] == seeker else "assistant", m["text"]) for m in c["messages"]]
            liked = {q["movieId"]: q["liked"] for q in (c.get("respondentQuestions") or []) if q.get("suggested")}
            yield from from_turns("redial", "recommending", turns, {"dialogue": c["conversationId"], "liked": liked})


SOURCES = {"wildfb": wildfb, "wildfeedback": wildfeedback, "multiwoz": multiwoz, "taskmaster": taskmaster,
           "redial": redial, "ccpe": ccpe}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("lab/data/sense2/raw.jsonl"))
    p.add_argument("--only", nargs="*", default=list(SOURCES))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from lab.sense.status import Status

    status = Status(Path(args.out).parent / "status.json", remote="sense2/status.json")
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with args.out.open("w") as f:
        for name in args.only:
            status.update(stage=f"extracting {name}")
            # Reservoir sample up to the source's cap, so big sources are read once without filling memory.
            keep, seen = [], 0
            for row in SOURCES[name]():
                seen += 1
                if len(keep) < CAPS[name]:
                    keep.append(row)
                elif (j := rng.randrange(seen)) < CAPS[name]:
                    keep[j] = row
            for row in keep:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[name] = {"seen": seen, "kept": len(keep), "ended": sum(r["said"] == "" for r in keep),
                            "tasks": dict(Counter(r["task"] for r in keep))}
            status.update(data={"sources": counts})
            print(name, counts[name], flush=True)
    status.update(stage="extracted", force=True)


if __name__ == "__main__":
    main()

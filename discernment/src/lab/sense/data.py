"""Build the feedback sense's datasets.

Training data is real: WildFB (THU-KEG, MIT licence), user follow-ups from WildChat labelled on a 4-level
satisfaction scale, mapped onto ours (1 rejection -> 0, 2 correction -> 20, 3 positive engagement -> 80,
4 thanks or praise -> 100). WildFB has no unclear class, so a small set of silent and non-committal replies is
added as null, with separate phrase lists for training and testing.

Shopping is never trained on. It is the unseen test domain, from the lives' diaries (E2/E4): the simulated
user's words, labelled by what they say (perfect -> 100, close -> 20, wrong kind or no purchase -> 0,
silence -> null), with the hidden true reward kept beside each row.

  uv run python -m lab.sense.data --out lab/data/sense --diaries lab/data/lives-gcp
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

WILDFB = "THU-KEG/WildFB"
WILDFB_LABEL = {1: "0", 2: "20", 3: "80", 4: "100"}
PER_LABEL = 12000  # training rows kept per WildFB label, so no class swamps the others
NULL_TRAIN = 6000
REQUEST_CHARS, SAID_CHARS = 400, 600  # long code pastes are cut; the end of a request and the start of a reply matter most

# Unclear replies: the words do not say whether the attempt worked. Train and test lists do not overlap.
NULL_TRAIN_PHRASES = ["hmm", "ok", "okay", "I guess", "whatever", "...", "hm.", "sure", "fine I suppose", "eh",
                      "idk", "let me think", "maybe", "right", "uh", "so", "well", "k", "interesting", "meh"]
NULL_TEST_PHRASES = ["mmm", "alright then", "if you say so", "huh", "I see", "oh", "yeah maybe", "noted", "hmmm ok",
                     "we'll see"]
SHOP_LABEL = {"perfect": "100", "close": "20", "wrong_kind": "0", "no_purchase": "0", "silent": "null"}


def english(text: str) -> bool:
    return bool(text) and sum(c.isascii() for c in text) / len(text) > 0.9


def cut(text: str, n: int, keep_end: bool = False) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else ("..." + text[-n:] if keep_end else text[:n] + "...")


def wildfb_rows(split: str):
    """(request, said, label) for each English WildFB row of a split."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WILDFB, f"{split}.jsonl", repo_type="dataset")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            users = [m["content"] for m in r["messages"] if m["role"] == "user"]
            feedback = r["user_feedback"]
            said = feedback["content"] if isinstance(feedback, dict) else json.loads(feedback)["content"]
            if not users or r["label"] not in WILDFB_LABEL or not english(users[-1] + said):
                continue
            yield cut(users[-1], REQUEST_CHARS, keep_end=True), cut(said, SAID_CHARS), WILDFB_LABEL[r["label"]]


def null_rows(requests: list[str], phrases: list[str], n: int, rng: random.Random):
    for _ in range(n):
        kind = rng.random()
        if kind < 0.35:
            yield {"request": rng.choice(requests), "next_move": "said nothing", "said": "", "label": "null"}
        elif kind < 0.5:
            yield {"request": rng.choice(requests), "next_move": "left the session", "said": "", "label": "null"}
        else:
            yield {"request": rng.choice(requests), "next_move": "replied", "said": rng.choice(phrases), "label": "null"}


def shopping_rows(diaries: Path):
    """The simulated user's reactions in the lives' diaries, with the hidden true reward."""
    for diary in sorted(diaries.glob("*/diary.jsonl")):
        for line in diary.open():
            d = json.loads(line)
            if d.get("split") not in ("practice", "retry") or d.get("reaction") not in SHOP_LABEL:
                continue
            yield {"request": cut(d["instruction"], REQUEST_CHARS, keep_end=True),
                   "next_move": "said nothing" if not d["said"] else "replied", "said": d["said"] or "",
                   "label": SHOP_LABEL[d["reaction"]], "reward": d["reward_hidden"], "life": diary.parent.name}


def write(path: Path, rows: list[dict]) -> dict:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "labels": dict(Counter(r["label"] for r in rows))}


def make_probe(run: Path, out: Path, seed: int = 0) -> dict:
    """The fixed probe set the E5 lives re-score each exam day: 300 reactions from the held-out tests (100 real
    WildFB reactions across the four scores, 50 unseen unclear phrases, 150 shopping reactions across its labels)."""
    rng = random.Random(seed)
    wild = [json.loads(line) for line in (run / "test_wildfb.jsonl").open()]
    shop = [json.loads(line) for line in (run / "test_shopping.jsonl").open()]
    pick = lambda rows, lab, n: rng.sample([r for r in rows if r["label"] == lab], min(n, sum(r["label"] == lab for r in rows)))
    rows = [r for lab in ("0", "20", "80", "100") for r in pick(wild, lab, 25)] + pick(wild, "null", 50)
    rows += [r for lab in ("0", "20", "100", "null") for r in pick(shop, lab, 38)]
    return write(out, rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("lab/data/sense"))
    p.add_argument("--diaries", type=Path, default=Path("lab/data/lives-gcp"))
    p.add_argument("--shop-per-label", type=int, default=1500, help="shopping test rows kept per label")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-from", type=Path, help="only write probe.jsonl from a finished run's test sets")
    args = p.parse_args()
    if args.probe_from:
        print(make_probe(args.probe_from, args.out / "probe.jsonl", args.seed))
        return
    from lab.sense.status import Status

    status = Status()
    status.update(stage="building data")
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Reservoir-sample each label, so the 3 GB file is read once without holding it in memory.
    keep, seen = {lab: [] for lab in WILDFB_LABEL.values()}, Counter()
    for request, said, label in wildfb_rows("train"):
        seen[label] += 1
        row = {"request": request, "next_move": "replied", "said": said, "label": label}
        if len(keep[label]) < PER_LABEL:
            keep[label].append(row)
        elif (j := rng.randrange(seen[label])) < PER_LABEL:
            keep[label][j] = row
        if sum(seen.values()) % 20000 == 0:
            status.update(data={"wildfb_read": sum(seen.values())})
    rows = [r for lab in keep.values() for r in lab]
    requests = [r["request"] for r in rows]
    rows += list(null_rows(requests, NULL_TRAIN_PHRASES, NULL_TRAIN, rng))
    rng.shuffle(rows)
    val, train = rows[:2000], rows[2000:]

    test = [{"request": q, "next_move": "replied", "said": s, "label": lab} for q, s, lab in wildfb_rows("test")]
    test += list(null_rows(requests, NULL_TEST_PHRASES, 500, rng))
    shop = list(shopping_rows(args.diaries))
    by_label = {lab: [r for r in shop if r["label"] == lab] for lab in set(SHOP_LABEL.values())}
    shop = [r for lab in by_label.values() for r in rng.sample(lab, min(len(lab), args.shop_per_label))]

    sets = {name: write(args.out / f"{name}.jsonl", data) for name, data in
            (("train", train), ("val", val), ("test_wildfb", test), ("test_shopping", shop))}
    sets["wildfb_english_seen"] = dict(seen)
    status.update(stage="data ready", data=sets)
    print(json.dumps(sets, indent=2))


if __name__ == "__main__":
    main()

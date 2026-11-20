"""What the tau^2-bench customer's last message looked like in the lessons of the sense v3 lives (Section 5.5).

  python3 src/tau2_reactions.py [data dir]

Counts the sense v3 lessons of Experiment 5, how many were failed attempts, and in how many of those failures the
customer's last message contained thanks, praise or a sign-off (found by keyword). It prints the failed lessons that
matched none of the keywords, so that they can be read: most are short confirmations, and about half are messages
from the middle of a request that the sense should have refused.
"""
import csv, gzip, re, sys
from pathlib import Path

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
lives = {r["life"] for r in csv.DictReader(open(DATA / "lives.csv")) if r["paper"] == "Exp 5" and r["judge"].startswith("sense")}
KEYWORDS = re.compile(
    r"thank|appreciate|grateful|cheers|thx"  # thanks
    r"|perfect|great|wonderful|excellent|awesome|sounds good|sounds correct|that works|exactly what"  # praise
    r"|bye|take care|that's all|that is all|all set|everything i (?:needed|need)|that's everything"  # sign-off
    r"|that should be everything|that covers everything|have a (?:good|great|nice)|that's it|done here|we're done",
    re.I)
lessons = [a for a in csv.DictReader(gzip.open(DATA / "attempts.csv.gz", "rt", encoding="utf-8"))
           if a["life"] in lives and a["lesson"] == "yes"]
failed = [a for a in lessons if float(a["score"]) < 100]
matched = [a for a in failed if KEYWORDS.search(a["user_said"])]
print(f"sense v3 lessons {len(lessons)}, failed {len(failed)} ({100 * len(failed) / len(lessons):.0f}%)")
print(f"failed lessons whose last customer message has thanks, praise or a sign-off: {len(matched)} of {len(failed)} "
      f"({100 * len(matched) / len(failed):.1f}%)")
print("\nfailed lessons with none of them:")
for a in failed:
    if a not in matched:
        print(" -", a["user_said"].replace("\n", " ")[:200], "|", a["verdict"][:20])

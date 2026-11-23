"""E18: keep one answer per question in bfcl's result files. Two copies of the exam once ran at the same time on one VM
(a launch that seemed cut off had started after all) and both appended their answers, so bfcl's scorer refused the
files (more answers than questions). Both copies answered from the same final state with near-greedy decoding; the
first answer to each question is kept, in bfcl's order.

    python -m lab.e18.dedupe <bfcl project root>   (e.g. ~/bfcl-e18/e17-qwen3-1.7b-teach-s15)
"""
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    for f in sorted(root.glob("result/**/*_result.json")):
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        seen, kept = set(), []
        for l in lines:
            i = json.loads(l)["id"]
            if i not in seen:
                seen.add(i)
                kept.append(l)
        if len(kept) < len(lines):
            f.write_text("\n".join(kept) + "\n")
            print(f"{f.relative_to(root)}: {len(lines)} -> {len(kept)}")


if __name__ == "__main__":
    main()

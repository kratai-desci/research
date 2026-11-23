"""E13: run bfcl on the practice set instead of its own multi-turn base questions.

    E13_DATA=<practice dir> python -m lab.e13.practice_bfcl generate --model Qwen/Qwen3-1.7B-FC --test-category multi_turn_base ...
    E13_DATA=<practice dir> python -m lab.e13.practice_bfcl evaluate --model Qwen/Qwen3-1.7B-FC --test-category multi_turn_base

<practice dir> is lab/e13/practice.py's output. bfcl reads every question and answer file through two paths in
bfcl_eval.utils; this points both at a copy of bfcl's data folder whose multi_turn_base files are the practice set, so
the handlers, the tool simulators and the scorer are bfcl's own, unchanged. Only multi_turn_base is redirected.
"""
import os
import shutil
import sys
from pathlib import Path

import bfcl_eval
import bfcl_eval.utils as U

NAME = "BFCL_v4_multi_turn_base.json"


def data_dir(practice: Path) -> Path:
    """bfcl's data folder with the practice set in place of multi_turn_base (built once next to the practice set)."""
    dst = practice / "bfcl-data"
    if not (dst / NAME).exists():
        shutil.copytree(Path(bfcl_eval.__file__).parent / "data", dst, dirs_exist_ok=True)
        shutil.copy(practice / NAME, dst / NAME)
        shutil.copy(practice / "possible_answer" / NAME, dst / "possible_answer" / NAME)
    return dst


if __name__ == "__main__":
    if os.environ.get("E13_DATA"):  # unset: bfcl's own questions (e.g. to re-score them the same way)
        d = data_dir(Path(os.environ["E13_DATA"]).resolve())
        U.PROMPT_PATH, U.POSSIBLE_ANSWER_PATH = d, d / "possible_answer"
    if os.environ.get("E13_STATE_ONLY"):
        # diagnosis only: score by bfcl's final-state check alone, dropping its check that the model's tool results
        # include every result the reference got (which fails a model for skipping a lookup the task did not need)
        import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker as C
        C.response_checker = lambda *a, **k: {"valid": True}
    from bfcl_eval.__main__ import cli
    sys.argv[0] = "bfcl"
    cli()

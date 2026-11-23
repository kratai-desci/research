"""E18: the official BFCL exam with a learned state from E15 (notes) or E17 (retrieved examples) in the prompt, the way
it was in practice. (E16's adapters need no wrapper: they are merged into the weights, lab/e18/merge.py, and examined
like E14's floors.)

    E18_NOTES=<notes.txt> python -m lab.e18.exam_bfcl generate --model <bfcl model> --test-category ...
    E18_MEMORY=<memory.jsonl> E18_TOKENIZER=<model path or id> [E18_BUDGET=1200] [E18_K=3] \
        python -m lab.e18.exam_bfcl generate ...
    python -m lab.e18.exam_bfcl evaluate --model <bfcl model> --test-category ...

bfcl's own CLI, handlers and checkers, unchanged; only the questions it loads for generating answers get one system
message at the start, exactly as in practice: E15's notes under E15's header; for E17, up to E18_K examples retrieved from
the final memory (logged tasks with the same tools, closest in meaning, within E18_BUDGET tokens) under E17's header. A
question whose first turn already opens with a system message gets the text added at the start of that message instead
(some chat templates allow one system message only). Single-turn and memory questions have no logged tasks with the same
tools, so E17 gives them no examples; E15's notes go into every question, as they did in every practice task.
"""
import os
import sys

import bfcl_eval._llm_response_generation as G
import bfcl_eval.utils as U

from lab.e15.notes import NOTES_HEADER

_load = U.load_dataset_entry


def _system_text():
    """A function from a question to the text to put in front of it (None: nothing)."""
    if os.environ.get("E18_NOTES"):
        notes = [l for l in open(os.environ["E18_NOTES"]).read().splitlines() if l.strip()]
        text = NOTES_HEADER + "\n" + "\n".join(notes) if notes else ""
        return lambda q: text
    if os.environ.get("E18_MEMORY"):
        from pathlib import Path

        from transformers import AutoTokenizer

        from lab.e17.rag import Embedder, Memory
        tok = AutoTokenizer.from_pretrained(os.environ["E18_TOKENIZER"], trust_remote_code=True)
        ntok = lambda t: len(tok.encode(t, add_special_tokens=False))
        mem = Memory(Embedder())
        mem.load(Path(os.environ["E18_MEMORY"]))
        k, budget = int(os.environ.get("E18_K", "3")), int(os.environ.get("E18_BUDGET", "1200"))

        def retrieve(q):
            if "involved_classes" not in q:  # single-turn questions: no logged task shares their tools
                return ""
            return mem.retrieve(q, k, budget, ntok)[0]
        return retrieve
    return None


def _add(entries, text_for):
    for e in entries:
        text = text_for(e)
        if not text:
            continue
        first = e["question"][0]
        if first and first[0].get("role") == "system":
            first[0]["content"] = text + "\n\n" + first[0]["content"]
        else:
            first.insert(0, {"role": "system", "content": text})
    return entries


if __name__ == "__main__":
    if os.environ.get("E18_REGISTER_E14"):  # the two newest models' handlers (lab/e14/new_handlers.py)
        from lab.e14.new_handlers import register
        register()
    text_for = _system_text() if sys.argv[1:2] == ["generate"] else None
    if text_for is not None:
        def load(*a, **kw):
            return _add(_load(*a, **kw), text_for)
        G.load_dataset_entry = load  # the main path (bfcl generate)
        U.load_dataset_entry = load  # the run-ids path
    from bfcl_eval.__main__ import cli
    sys.argv[0] = "bfcl"
    cli()

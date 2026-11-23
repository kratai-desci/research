"""E15: skill notes. A frozen tiny model does the E13 practice tasks one after another; after each task it reads its
attempt and the oracle's feedback, writes one short lesson into its notes, and carries the notes in its prompt on the
next task. The weights never change.

Arms:
  frozen  no notes (the practice-pool baseline; tasks run in parallel, since nothing carries over)
  teach   oracle teaching: pass or fail, the checker's reason, and the expected calls
  judge   oracle judging: pass or fail only

Each model runs through its own BFCL handler (the same prompt and tool-call format as on the BFCL exam), against a vLLM
server that is already up (deploy/gcp/run_e15.sh starts it, then several runs of the same model share it). The notes
enter as a system message at the start of the task; bfcl puts its own system prompt with the tools in front of it.
Scoring is BFCL's own multi-turn checker against E13's hidden reference.

    python -m lab.e15.notes --registry Qwen/Qwen3-1.7B-FC --data <v3> <v4> <v5> --out <dir> --arm teach \
        [--model-path <local weights>] [--tasks 2400] [--seed 15] [--cap 600]

Writes <out>/log.jsonl (one line per task), <out>/notes.txt (the current notes), <out>/notes-<n>.txt (a snapshot every
100 tasks) and <out>/status.json; resumes from <out>/state.json after a crash.
"""
import argparse
import copy
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
from bfcl_eval.utils import populate_test_cases_with_predefined_functions

THINK = re.compile(r"<think>.*?</think>", re.S)
STRIP = " -*•\t"  # bullet marks and spaces around a note
NOTES_HEADER = "Notes you wrote from earlier tasks with these tools (use them where they apply):"


def _words(note: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", note.lower()))


def is_repeat(lesson: str, notes: list[str], threshold: float = 0.6) -> bool:
    """A lesson that mostly repeats a note already kept (word overlap, Jaccard, at least the threshold)."""
    w = _words(lesson)
    return any(len(w & _words(n)) / max(1, len(w | _words(n))) >= threshold for n in notes)


def build_handler(registry: str, model_path: str | None):
    if registry.startswith("e14-"):  # the two newest models' handlers (lab/e14/new_handlers.py)
        from lab.e14.new_handlers import register
        register()
    config = MODEL_CONFIG_MAPPING[registry]
    handler = config.model_handler(model_name=config.model_name, temperature=0.001, registry_name=registry,
                                   is_fc_model=config.is_fc_model)
    # the server is already up (run_e15.sh); this loads the tokenizer and waits for it
    handler.spin_up_local_server(num_gpus=1, gpu_memory_utilization=0.9, backend="vllm", skip_server_setup=True,
                                 local_model_path=model_path)
    return handler


SPLIT = Path(__file__).resolve().parents[3] / "data" / "split.json"  # the fixed practice and held-out tasks, shared by every experiment


def load_pool(dirs: list[str], seed: int, split: Path | None = SPLIT) -> list[tuple]:
    """The E13 pool as (set, question, reference). With the split file (the default): its practice tasks in its order,
    then its held-out tasks, so every run practises on the same tasks; a seed other than the split's reorders the
    practice tasks only. Without it: the whole pool shuffled with the seed (how the split was made)."""
    pool = []
    for d in dirs:
        d = Path(d)
        qs = [json.loads(l) for l in open(d / "BFCL_v4_multi_turn_base.json")]
        ref = d / "possible_answer/BFCL_v4_multi_turn_base.json"
        gts = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(ref)} if ref.exists() else {}
        pool += [(d.name, q, gts.get(q["id"])) for q in qs]
    if split is None or not Path(split).exists():
        random.Random(seed).shuffle(pool)
        return pool
    sp = json.loads(Path(split).read_text())
    by = {(src, q["id"]): (src, q, gt) for src, q, gt in pool}
    practice = [by[tuple(k)] for k in sp["practice"]]
    if seed != sp["seed"]:
        random.Random(seed).shuffle(practice)
    return practice + [by[tuple(k)] for k in sp["heldout"]]


def make_split(dirs: list[str], seed: int = 15, n: int = 1200) -> dict:
    """The split: the pool shuffled with the seed; the first n tasks to practise on (in that order), the rest held out."""
    pool = load_pool(dirs, seed, split=None)
    keys = [[src, q["id"]] for src, q, _ in pool]
    return {"seed": seed, "practice": keys[:n], "heldout": keys[n:]}


class Runner:
    def __init__(self, a):
        self.a = a
        self.handler = build_handler(a.registry, a.model_path)
        self.tok = self.handler.tokenizer

    # ---- the model, as a plain chat model (for lessons and curation) ----
    def chat(self, text: str, max_tokens: int, seed: int) -> str:
        msgs = [{"role": "user", "content": text}]
        try:
            prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        r = self.handler.client.completions.create(model=self.handler.model_path_or_id, prompt=prompt,
                                                   max_tokens=max_tokens, temperature=self.a.temp, seed=seed)
        return THINK.sub("", r.choices[0].text).strip()

    def ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    # ---- one task ----
    def solve(self, i: int, q: dict, gt: list, notes: list[str]) -> dict:
        entry = {"id": f"multi_turn_base_{i}", "question": copy.deepcopy(q["question"]),
                 "initial_config": q.get("initial_config", {}), "involved_classes": q["involved_classes"]}
        populate_test_cases_with_predefined_functions([entry])
        if notes:
            entry["question"][0].insert(0, {"role": "system", "content": NOTES_HEADER + "\n" + "\n".join(notes)})
        t0 = time.time()
        try:
            result, _ = self.handler.inference(copy.deepcopy(entry), include_input_log=False, exclude_state_log=True)
        except Exception as e:  # a failed request counts as a wrong answer, as on the exam
            result = f"Error during inference: {e}"
        verdict = _evaluate_single_multi_turn_entry(self.handler, entry["id"], result, gt, copy.deepcopy(entry),
                                                    self.a.registry, "multi_turn_base")
        return {"entry": entry, "result": result, "valid": bool(verdict["valid"]),
                "error": None if verdict["valid"] else verdict.get("error"), "secs": time.time() - t0}

    def decoded(self, result) -> list[list[str]]:
        """The model's calls per turn, as call strings (unparseable steps left out, as the checker does)."""
        turns = []
        for turn in result if isinstance(result, list) else []:
            calls = []
            for step in turn:
                try:
                    calls += [c for c in self.handler.decode_execute(step, has_tool_call_tag=False) if c]
                except Exception:
                    if isinstance(step, str) and step.strip():
                        calls.append(f"(text, no call) {step.strip()[:150]}")
            turns.append(calls)
        return turns

    def trace(self, i: int, entry: dict, result, gt: list, teach: bool) -> str:
        """The task as the model saw it, its calls with the tools' replies, and (teaching only) the expected calls."""
        cfg, classes = entry["initial_config"], entry["involved_classes"]
        out = []
        calls_by_turn = self.decoded(result)
        for t, turn in enumerate(entry["question"]):
            user = " ".join(m["content"] for m in turn if m["role"] == "user")
            out.append(f"Turn {t + 1}. User: {user[:400]}")
            calls = [c for c in (calls_by_turn[t] if t < len(calls_by_turn) else []) if not c.startswith("(text")]
            texts = [c for c in (calls_by_turn[t] if t < len(calls_by_turn) else []) if c.startswith("(text")]
            if calls:
                try:  # replay the calls on a fresh copy of the tools to show what each one returned
                    replies, _ = execute_multi_turn_func_call(calls, cfg, classes, "e15replay", f"r{i}_{t}")
                except Exception:
                    replies = [""] * len(calls)
                for c, r in zip(calls, replies):
                    out.append(f"  You called: {c}\n    It returned: {str(r)[:200]}")
            for x in texts:
                out.append(f"  You replied {x[16:]}")
            if not calls and not texts:
                out.append("  You made no call.")
            if teach and t < len(gt):
                out.append("  Expected calls: " + ("; ".join(gt[t]) if gt[t] else "none"))
        return "\n".join(out)

    def lesson(self, i: int, s: dict, gt: list, notes: list[str]) -> str:
        teach = self.a.arm == "teach"
        outcome = "You completed the task correctly." if s["valid"] else "You did NOT complete the task correctly."
        if teach and not s["valid"] and s["error"]:
            msg = s["error"].get("error_message")
            outcome += f" Checker: {str(msg)[:300]}"
        text = (f"You just did a task with tools.\n\n{self.trace(i, s['entry'], s['result'], gt, teach)}\n\n"
                f"Outcome: {outcome}\n\nYour current notes:\n" + ("\n".join(notes) if notes else "(none yet)") +
                "\n\nWrite ONE new lesson for your notes: a short, general rule (under 30 words) that will help you "
                "on future tasks with these tools. Do not repeat a note you already have. Reply with the lesson only.")
        line = self.chat(text, 80, self.a.seed * 100_000 + i).splitlines()
        line = next((l.strip(STRIP) for l in line if l.strip(STRIP)), "")
        return f"- {line}" if line else ""

    def curate(self, i: int, notes: list[str]) -> list[str]:
        """Merge and trim the notes back under the cap; the model does it, oldest notes dropped if it fails."""
        text = ("Here are your notes from past tasks with tools:\n" + "\n".join(notes) +
                f"\n\nRewrite them as at most {self.a.max_notes} short bullet points (each under 30 words), merging "
                "repeats and keeping the most useful general rules. Reply with the bullet points only, one per line, "
                "each starting with '- '.")
        out = self.chat(text, 600, self.a.seed * 100_000 + 50_000 + i)
        new = ["- " + l.strip(STRIP) for l in out.splitlines() if l.strip().startswith(("-", "*", "•")) and l.strip(STRIP)]
        new = new[: self.a.max_notes] if new else list(notes)
        while new and self.ntok("\n".join(new)) > self.a.cap:  # still too long: drop the oldest
            new.pop(0)
        return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True, help="bfcl model name (its handler and prompt format)")
    ap.add_argument("--model-path", default=None, help="local weights folder, if the server serves one")
    ap.add_argument("--data", required=True, nargs="+", help="practice set folders (E13 v3, v4, v5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["frozen", "teach", "judge"], required=True)
    ap.add_argument("--tasks", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=15, help="task order, and sampling when writing notes")
    ap.add_argument("--cap", type=int, default=600, help="notes size cap, in tokens")
    ap.add_argument("--max-notes", type=int, default=12, help="bullet points kept when the notes are curated")
    ap.add_argument("--temp", type=float, default=0.7, help="sampling temperature for lessons and curation")
    ap.add_argument("--split", default=str(SPLIT), help="the practice split file ('' for none)")
    ap.add_argument("--workers", type=int, default=8, help="frozen arm only: tasks in parallel")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(a.data, a.seed, Path(a.split) if a.split else None)[: a.tasks]
    run = Runner(a)

    st = {"done": 0, "notes": [], "curations": 0, "started": time.time()}
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        print(f"resumed after task {st['done']}", flush=True)
    log = [json.loads(l) for l in open(out / "log.jsonl")][: st["done"]] if (out / "log.jsonl").exists() else []
    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))  # drop lines past the last saved state

    def save(stage):
        tmp = out / "state.tmp"
        tmp.write_text(json.dumps(st))
        tmp.replace(out / "state.json")
        (out / "notes.txt").write_text("\n".join(st["notes"]))
        ok = [r["valid"] for r in log]
        (out / "status.json").write_text(json.dumps({
            "arm": a.arm, "registry": a.registry, "seed": a.seed, "stage": stage, "done": len(log), "tasks": len(pool),
            "success": round(100 * sum(ok) / len(ok), 1) if ok else None,
            "last50": round(100 * sum(ok[-50:]) / len(ok[-50:]), 1) if ok else None,
            "notes": len(st["notes"]), "notes_tokens": run.ntok("\n".join(st["notes"])), "curations": st["curations"],
            "secs_per_task": round(sum(r["secs"] for r in log) / len(log), 1) if log else None,
            "started": st["started"], "updated": time.time(),
            "settings": {k: v for k, v in vars(a).items() if k not in ("data", "out")}}))

    def record(i, src, q, s, lesson=""):
        r = {"i": i, "set": src, "id": q["id"], "valid": s["valid"], "secs": round(s["secs"], 1),
             "error": (s["error"] or {}).get("error_type"), "lesson": lesson, "notes_tokens": run.ntok("\n".join(st["notes"]))}
        log.append(r)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")

    save("running")
    todo = [(i, *p) for i, p in enumerate(pool) if i >= st["done"]]
    if a.arm == "frozen":  # nothing carries over between tasks: run them in parallel, record in order
        with ThreadPoolExecutor(a.workers) as ex:
            for k in range(0, len(todo), 50):
                chunk = todo[k: k + 50]
                for (i, src, q, gt), s in zip(chunk, ex.map(lambda t: run.solve(t[0], t[2], t[3], []), chunk)):
                    record(i, src, q, s)
                st["done"] = len(log)
                save("running")
    else:
        for i, src, q, gt in todo:
            s = run.solve(i, q, gt, st["notes"])
            t0 = time.time()
            lesson = run.lesson(i, s, gt, st["notes"])
            if lesson and not is_repeat(lesson, st["notes"]):
                st["notes"].append(lesson)
            if run.ntok("\n".join(st["notes"])) > a.cap:
                st["notes"] = run.curate(i, st["notes"])
                st["curations"] += 1
            s["secs"] += time.time() - t0
            record(i, src, q, s, lesson)
            st["done"] = len(log)
            if st["done"] % 100 == 0:
                (out / f"notes-{st['done']}.txt").write_text("\n".join(st["notes"]))
            save("running")
    save("done")


if __name__ == "__main__":
    main()

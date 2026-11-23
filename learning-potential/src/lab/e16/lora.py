"""E16: LoRA fine-tuning on the job. A tiny model does E15's practice tasks one after another through its own BFCL
handler and vLLM; every 10 tasks a LoRA adapter is trained on what the arm allows, and vLLM loads the new adapter
without restarting. The base weights never change.

Arms:
  teach  oracle teaching: the expected calls of every task, written in the model's own tool-call format (supervised
         fine-tuning)
  judge  oracle judging: the model's own attempts that passed, exactly as it wrote them (self-imitation)
  teach-think  (E19, Qwen3 only) oracle teaching with the reasoning kept: as teach, but each target opens with a short
         think block the base model writes itself, shown the expected reply as a hint (STaR's rationalization),
         instead of an empty one; the think block is trained on with the calls

Training examples are (prompt, answer) pairs exactly as the handler builds them. For judging, a recorder on the
handler's connection keeps every prompt and answer of a task. For teaching, the same recorder stands in for the model
and answers with the expected calls, so the handler builds the prompts (tool results included) the real model would
see after giving those answers.

    python -m lab.e16.lora --registry Qwen/Qwen3-1.7B-FC --data <v3> <v4> <v5> --out <dir> --arm teach \
        [--model-path <local weights>] [--tasks 1200] [--seed 15]

Writes <out>/log.jsonl, <out>/status.json, <out>/adapter-<n>/ every 100 tasks (for the probes) and resumes from
<out>/state.json (saved at every update) after a crash. Needs vLLM serving the model with LoRA enabled and runtime
adapter loading (deploy/gcp/run_e16.sh).
"""
import argparse
import ast
import copy
import json
import random
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import requests
import torch
from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
from bfcl_eval.utils import populate_test_cases_with_predefined_functions

from lab.e15.notes import SPLIT, build_handler, load_pool

DONE = "Done."  # the closing reply of a turn in the teaching targets (a turn ends on a reply with no call)


# ---- the expected calls in each model's own tool-call format ----
def parse_call(s: str) -> tuple[str, dict]:
    """BFCL's reference call string, e.g. "cd(folder='docs')", as (name, arguments)."""
    node = ast.parse(s.strip(), mode="eval").body
    name = ast.unparse(node.func)
    return name, {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def _pythonic(calls) -> str:
    return "[" + ", ".join(f"{n}(" + ", ".join(f"{k}={v!r}" for k, v in a.items()) + ")" for n, a in calls) + "]"


def _xml_value(v) -> str:
    return v if isinstance(v, str) else json.dumps(v)


RENDER = {  # bfcl model -> how that model writes a list of calls
    "Qwen/Qwen3-1.7B-FC": lambda calls: "\n".join(
        "<tool_call>\n" + json.dumps({"name": n, "arguments": a}) + "\n</tool_call>" for n, a in calls),
    "tiiuae/Falcon3-1B-Instruct-FC": lambda calls: "<tool_call>\n" + json.dumps(
        [{"name": n, "arguments": a} for n, a in calls]) + "\n</tool_call>",
    "meta-llama/Llama-3.2-1B-Instruct-FC": _pythonic,
    "google/gemma-3-1b-it": _pythonic,
    "e14-lfm2.5-1.2b": lambda calls: "<|tool_call_start|>" + _pythonic(calls) + "<|tool_call_end|>",
    "e14-qwen3.5-0.8b": lambda calls: "\n".join(
        "<tool_call>\n<function=" + n + ">\n" + "".join(f"<parameter={k}>\n{_xml_value(v)}\n</parameter>\n"
                                                         for k, v in a.items()) + "</function>\n</tool_call>"
        for n, a in calls),
}


# Prefixes a model's own answers always start with. Qwen3 reasons in a <think> block first; the teaching targets give
# it an empty one (Qwen's own form for answering without reasoning), which the trainer moves into the masked prompt, so
# the targets teach the calls without teaching the model to stop reasoning.
PREFIX = {"Qwen/Qwen3-1.7B-FC": "<think>\n\n</think>\n\n"}
EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*")


class Recorder:
    """Stands in for the handler's OpenAI client: keeps every prompt and answer. With a script of answers, it gives
    those instead of asking the model (the teaching targets). With think set (E19), each scripted answer is a reply
    without its think block, and think(prompt, reply) writes the full answer, reasoning included."""

    def __init__(self, real):
        self.real, self.script, self.calls, self.think = real, None, [], None

    @property
    def completions(self):
        return self

    def create(self, **kw):
        if self.script is not None:
            text = self.script.pop(0) if self.script else DONE
            if self.think is not None:
                text = self.think(kw["prompt"], text)
            resp = SimpleNamespace(choices=[SimpleNamespace(text=text)],
                                   usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))
        else:
            resp = self.real.completions.create(**kw)
            text = resp.choices[0].text
        self.calls.append((kw["prompt"], text))
        return resp


# ---- E19: the reasoning kept in the teaching targets ----
ASSISTANT = "<|im_start|>assistant\n"
HINT = ("(A note for this step only, which you will not see next time: the correct next reply is shown below. Think "
        "briefly, in a few sentences, about why it is the right next step given the conversation so far, as if you "
        "worked it out yourself. Do not mention this note or that you were shown the reply.)\n\nCorrect next reply:\n")
DONE_HINT = "No more function calls: every request of this turn is complete, so reply briefly that it is done."
LEAK = re.compile(r"\b(note|hint|shown|given (the )?(correct|reply))\b", re.I)


class Rationalizer:
    """Writes the think block of a teaching target with the base model (vLLM's base weights, not the adapter), given
    the expected reply as a hint, and drops any sentence that mentions the hint."""

    def __init__(self, client, model: str, max_tokens: int = 512):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.n = self.leaked = self.cut = 0

    def __call__(self, prompt: str, reply: str) -> str:
        bare = EMPTY_THINK.sub("", reply)
        shown = DONE_HINT if bare.strip() == DONE else bare
        if not prompt.endswith(ASSISTANT):
            raise ValueError("unexpected prompt end for a think target")
        p = prompt[: -len(ASSISTANT)] + "<|im_start|>user\n" + HINT + shown + "<|im_end|>\n" + ASSISTANT + "<think>\n"
        r = self.client.completions.create(model=self.model, prompt=p, temperature=0.001, max_tokens=self.max_tokens,
                                           stop=["</think>"])
        text, self.n = r.choices[0].text, self.n + 1
        if r.choices[0].finish_reason == "length":  # cut at the last full sentence
            self.cut += 1
            text = text[: max(text.rfind(". "), text.rfind(".\n")) + 1] or text
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        kept = [x for x in parts if not LEAK.search(x)]
        self.leaked += int(len(kept) < len(parts))
        return "<think>\n" + " ".join(kept).strip() + "\n</think>\n\n" + bare


# ---- the LoRA trainer ----
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Trainer:
    def __init__(self, path: str, rank: int, lr: float, max_len: int):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        names = {n.split(".")[-1] for n, _ in model.named_modules()}
        targets = [t for t in LORA_TARGETS if t in names]
        self.model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                                      target_modules=targets, task_type="CAUSAL_LM"))
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr)
        self.max_len, self.skipped = max_len, 0
        self.end = self._turn_end()

    def _turn_end(self) -> str:
        """What the chat template writes after an assistant reply (e.g. <|im_end|>), so the model learns to stop."""
        try:
            t = self.tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": "ZQX"}],
                                             tokenize=False)
            end = t.split("ZQX", 1)[1]
            return end.rstrip("\n") or (self.tok.eos_token or "")
        except Exception:
            return self.tok.eos_token or ""

    def train(self, examples: list[tuple[str, str]], accum: int = 4) -> float | None:
        """One pass over the examples (batch 1, gradients accumulated); returns the mean loss."""
        self.model.train()
        losses, n = [], 0
        self.opt.zero_grad()
        for prompt, answer in examples:
            m = EMPTY_THINK.match(answer)  # an empty think block is context, not something to learn
            if m:
                prompt, answer = prompt + answer[: m.end()], answer[m.end():]
            p = self.tok(prompt, add_special_tokens=False)["input_ids"]
            a = self.tok(answer + self.end, add_special_tokens=False)["input_ids"]
            if len(p) + len(a) > self.max_len:
                self.skipped += 1
                continue
            ids = torch.tensor([p + a], device="cuda")
            try:
                # only the answer's positions are scored: the full vocabulary scores of a long prompt would not fit
                logits = self.model(input_ids=ids, logits_to_keep=len(a) + 1).logits[0, :-1].float()
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(a, device="cuda"))
                (loss / accum).backward()
            except torch.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                self.skipped += 1
                continue
            losses.append(loss.item())
            n += 1
            if n % accum == 0:
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.opt.step()
                self.opt.zero_grad()
        if n % accum:
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.opt.zero_grad()
        self.model.eval()
        return sum(losses) / len(losses) if losses else None

    def save(self, d: Path):
        tmp = d.with_name(d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        self.model.save_pretrained(tmp)
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)

    def load(self, d: Path, opt: Path | None):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(self.model, load_file(str(d / "adapter_model.safetensors")))
        if opt and opt.exists():
            self.opt.load_state_dict(torch.load(opt, map_location="cuda"))


# ---- one run ----
class Runner:
    def __init__(self, a):
        self.a = a
        self.handler = build_handler(a.registry, a.model_path)
        self.rec = Recorder(self.handler.client)
        self.handler.client = self.rec
        self.base = self.handler.model_path_or_id  # the served base model's name
        self.render = RENDER[a.registry]
        self.prefix = PREFIX.get(a.registry, "")
        self.think = None
        if a.arm == "teach-think":
            if a.registry not in PREFIX:
                raise SystemExit("teach-think is for models that reason in a think block (Qwen3)")
            self.think = Rationalizer(self.rec.real, self.base)

    def entry(self, i: int, q: dict) -> dict:
        e = {"id": f"multi_turn_base_{i}", "question": copy.deepcopy(q["question"]),
             "initial_config": q.get("initial_config", {}), "involved_classes": q["involved_classes"]}
        populate_test_cases_with_predefined_functions([e])
        return e

    def run(self, e: dict, gt: list, script: list[str] | None):
        self.rec.calls, self.rec.script = [], script
        try:
            result, _ = self.handler.inference(copy.deepcopy(e), include_input_log=False, exclude_state_log=True)
        except Exception as ex:  # a failed request counts as a wrong answer, as on the exam
            result = f"Error during inference: {ex}"
        finally:
            self.rec.script = None
        v = _evaluate_single_multi_turn_entry(self.handler, e["id"], result, gt, copy.deepcopy(e), self.a.registry,
                                              "multi_turn_base")
        return bool(v["valid"]), (None if v["valid"] else (v.get("error") or {}).get("error_type")), list(self.rec.calls)

    def solve(self, i: int, q: dict, gt: list):
        return self.run(self.entry(i, q), gt, None)

    def teaching(self, i: int, q: dict, gt: list):
        """The expected calls as the model's answers. Returns (pairs, the replay passes the checker, the rendering
        decodes back to the expected calls; not checked for our own handlers, which parse while querying)."""
        script, roundtrip = [], not self.a.registry.startswith("e14-")  # our handlers parse while querying
        for turn in gt:
            if turn:
                calls = [parse_call(c) for c in turn]
                text = self.render(calls)
                try:
                    back = [parse_call(c) for c in self.handler.decode_execute(text, has_tool_call_tag=False)]
                except Exception:
                    back = None
                if not self.a.registry.startswith("e14-"):
                    roundtrip &= back == calls
                script.append(self.prefix + text)
            script.append(self.prefix + DONE)
        self.rec.think = self.think
        try:
            valid, _, pairs = self.run(self.entry(500_000 + i, q), gt, script)
        finally:
            self.rec.think = None
        # the closing replies are trained on too: they teach the model to end a turn once the calls are made (without
        # them, the first pilot learnt to keep calling until the turn was cut off)
        return pairs, valid, roundtrip

    def use_adapter(self, name: str, path: Path | None):
        """Point the handler at a new adapter in vLLM (None: the base model) and unload the previous one."""
        root = self.handler.base_url.rstrip("/")
        old = self.handler.model_path_or_id
        if path is not None:
            r = requests.post(f"{root}/load_lora_adapter", json={"lora_name": name, "lora_path": str(path)}, timeout=600)
            r.raise_for_status()
            self.handler.model_path_or_id = name
        else:
            self.handler.model_path_or_id = self.base
        if old != self.base and old != self.handler.model_path_or_id:
            requests.post(f"{root}/unload_lora_adapter", json={"lora_name": old}, timeout=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["teach", "judge", "teach-think"], required=True)
    ap.add_argument("--tasks", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--every", type=int, default=10, help="tasks between updates")
    ap.add_argument("--replay", type=int, default=2, help="earlier examples per new example in an update")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=12288, help="longer examples are skipped")
    ap.add_argument("--check", type=int, default=0, help="only build the teaching targets of this many tasks and "
                                                         "report them (no training)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(a.data, a.seed, Path(a.split) if a.split else None)[: a.tasks]
    run = Runner(a)

    if a.check:  # hand check of the teaching targets: they must decode back and replay as correct
        rows = []
        for i, (src, q, gt) in enumerate(pool[: a.check]):
            pairs, valid, rt = run.teaching(i, q, gt)
            rows.append({"i": i, "id": q["id"], "set": src, "replay_valid": valid, "roundtrip": rt, "steps": len(pairs),
                         "first_answer": pairs[0][1] if pairs else None, "prompt_tail": pairs[0][0][-300:] if pairs else None})
        (out / "check.json").write_text(json.dumps(rows, indent=1))
        print(json.dumps({"tasks": len(rows), "replay_valid": sum(r["replay_valid"] for r in rows),
                          "roundtrip": sum(r["roundtrip"] for r in rows)}), flush=True)
        return

    trainer = Trainer(a.model_path or run.base, a.rank, a.lr, a.max_len)
    st = {"done": 0, "updates": 0, "started": time.time(), "last_loss": None, "teach_replay_valid": 0,
          "teach_roundtrip": 0}
    if (out / "state.json").exists():
        st = json.loads((out / "state.json").read_text())
        if (out / "adapter-latest" / "adapter_model.safetensors").exists():
            trainer.load(out / "adapter-latest", out / "opt.pt")
            run.use_adapter(f"a{st['updates']}", out / "adapter-latest")
        # else nothing has been learned yet (e.g. judging with no passed task): the base model, as before the stop
        print(f"resumed after task {st['done']}", flush=True)
    log = []
    if (out / "log.jsonl").exists():  # up to the last saved update; a line cut off by a lost VM ends it
        for line in open(out / "log.jsonl"):
            try:
                log.append(json.loads(line))
            except json.JSONDecodeError:
                break
        log = log[: st["done"]]
    (out / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    old = []
    if (out / "examples.jsonl").exists():
        for line in open(out / "examples.jsonl"):
            try:
                old.append(tuple(json.loads(line)))
            except json.JSONDecodeError:  # a line cut off by a lost VM
                break
    rng = random.Random(a.seed * 1000 + st["done"])

    def status(stage):
        ok = [r["valid"] for r in log]
        (out / "status.json").write_text(json.dumps({
            "arm": a.arm, "registry": a.registry, "seed": a.seed, "stage": stage, "done": len(log), "tasks": len(pool),
            "success": round(100 * sum(ok) / len(ok), 1) if ok else None,
            "last50": round(100 * sum(ok[-50:]) / len(ok[-50:]), 1) if ok else None,
            "updates": st["updates"], "examples": len(old), "last_loss": st["last_loss"], "skipped": trainer.skipped,
            "teach_replay_valid": st["teach_replay_valid"], "teach_roundtrip": st["teach_roundtrip"],
            "think": {"written": run.think.n, "leaks_dropped": run.think.leaked, "cut": run.think.cut} if run.think else None,
            "secs_per_task": round(sum(r["secs"] for r in log) / len(log), 1) if log else None,
            "started": st["started"], "updated": time.time(),
            "settings": {k: v for k, v in vars(a).items() if k not in ("data", "out")}}))

    status("running")
    new = []
    for i in range(st["done"], len(pool)):
        src, q, gt = pool[i]
        t0 = time.time()
        valid, err, pairs = run.solve(i, q, gt)
        if a.arm == "judge" and valid:
            new += pairs
        elif a.arm in ("teach", "teach-think"):
            tpairs, tvalid, rt = run.teaching(i, q, gt)
            st["teach_replay_valid"] += int(tvalid)
            st["teach_roundtrip"] += int(rt)
            new += tpairs
        r = {"i": i, "set": src, "id": q["id"], "valid": valid, "error": err, "secs": round(time.time() - t0, 1),
             "examples": len(new), "update": None}
        if (i + 1) % a.every == 0 or i == len(pool) - 1:
            t1 = time.time()
            if new:
                batch = new + rng.sample(old, min(len(old), a.replay * len(new)))
                rng.shuffle(batch)
                st["last_loss"] = trainer.train(batch)
                st["updates"] += 1
                trainer.save(out / "adapter-latest")
                torch.save(trainer.opt.state_dict(), out / "opt.pt")
                run.use_adapter(f"a{st['updates']}", out / "adapter-latest")
                with open(out / "examples.jsonl", "a") as f:
                    f.writelines(json.dumps(list(p)) + "\n" for p in new)
                old += new
                new = []
            r["update"] = {"loss": st["last_loss"], "secs": round(time.time() - t1, 1), "n": st["updates"]}
            r["secs"] += r["update"]["secs"]
        log.append(r)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        if (i + 1) % a.every == 0 or i == len(pool) - 1:  # the state is saved at every update
            st["done"] = len(log)
            if (i + 1) % 100 == 0:  # a snapshot for the probes
                if (out / "adapter-latest").exists():
                    shutil.copytree(out / "adapter-latest", out / f"adapter-{i + 1}", dirs_exist_ok=True)
                else:  # nothing learned yet (e.g. judging with no passed task): the base model
                    (out / f"adapter-{i + 1}").mkdir(exist_ok=True)
                    (out / f"adapter-{i + 1}" / "BASE").write_text("no adapter yet: the base model")
            tmp = out / "state.tmp"
            tmp.write_text(json.dumps(st))
            tmp.replace(out / "state.json")
        status("running")
    status("done")


if __name__ == "__main__":
    main()

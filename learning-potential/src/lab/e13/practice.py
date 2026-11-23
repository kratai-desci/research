"""E13: the practice ground. A bigger model writes multi-turn practice tasks for BFCL's own tool simulators, each with a
hidden reference solution. A task is kept only if its reference runs in the simulators without an error, it passes one
strict review, and a solver (the same model, through the simulators, never seeing the reference) completes it as BFCL
scores it. Each task follows the shape of BFCL's base group: turns per task and calls per turn are drawn from that group's
own counts (statistics only; the writer never sees BFCL's questions).

    python -m lab.e13.practice --n 800 --out <dir> [--workers 16] [--model gemini-3-flash-preview] [--status <json>]

Output, in BFCL's own format so its runner and scorer read it unchanged (lab/e13/practice_bfcl.py points bfcl at it):
    <dir>/BFCL_v4_multi_turn_base.json                   the questions (ids multi_turn_base_<i>)
    <dir>/possible_answer/BFCL_v4_multi_turn_base.json   the hidden reference solutions ("ground_truth")
    <dir>/rejected.jsonl                                  tasks dropped, with the reason
The writer sees only the tools: their descriptions (BFCL's function docs) and their simulator source, never BFCL's
exam questions. Tool sets are drawn at random, one third single and two thirds pairs, as in the exam's base group.
Runs inside bfcl-eval's venv (it imports the simulators).
"""
import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import bfcl_eval
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

from lab.vertex import Gemini

PKG = Path(bfcl_eval.__file__).parent
SRC = PKG / "eval_checker/multi_turn_eval/func_source_code"
CLASSES = {  # class: (function doc file, simulator source file)
    "GorillaFileSystem": ("gorilla_file_system", "gorilla_file_system"),
    "MathAPI": ("math_api", "math_api"),
    "MessageAPI": ("message_api", "message_api"),
    "TwitterAPI": ("posting_api", "posting_api"),
    "TicketAPI": ("ticket_api", "ticket_api"),
    "TradingBot": ("trading_bot", "trading_bot"),
    "TravelAPI": ("travel_booking", "travel_booking"),
    "VehicleControlAPI": ("vehicle_control", "vehicle_control"),
}
# BFCL v4 multi_turn_base, counted from its answer file: turns per task (2-6 kept) and calls per turn (1-4 kept)
TURNS = {2: 40, 3: 50, 4: 50, 5: 43, 6: 12}
CALLS = {1: 474, 2: 156, 3: 67, 4: 20}
# BFCL's design (v5): four main tool sets, 50 tasks each; the other four only join a main set as its companion. Counted
# from its question file: 65 tasks use a main set alone, 135 add one companion, in these numbers.
PRIMARY = ["GorillaFileSystem", "VehicleControlAPI", "TradingBot", "TravelAPI"]
COMPANION = {"MessageAPI": 40, "TwitterAPI": 39, "TicketAPI": 31, "MathAPI": 25}
ALONE = 65 / 200
HERE = Path(__file__).parent
THEMES = [
    "a small accounting office closing the month", "a freelance designer organising client work", "a student finishing a thesis",
    "a family planning a holiday", "a sales rep on the road", "a support agent handling complaints", "a day trader checking positions",
    "a project manager preparing a status report", "a researcher cleaning up experiment logs", "a car owner before a long drive",
    "an office admin tidying shared folders", "a small shop owner tracking orders", "a teacher preparing class material",
    "a social media manager for a cafe", "a traveller whose plans changed", "an IT person handling tickets", "a parent organising school forms",
    "a recruiter keeping in touch with candidates", "a journalist with a deadline", "a retiree managing investments",
    "a consultant billing several clients", "a commuter in a hurry", "a startup founder preparing for investors",
    "an event planner booking trips for a team", "a data analyst summarising numbers", "a writer backing up drafts",
]

SYSTEM = """You write practice tasks for an AI assistant that works through software tools for a user.

You get the tools: their descriptions, as the assistant sees them, and the source code of the simulator that runs them.
Write ONE realistic multi-turn task, as JSON:

{"initial_config": {"<ClassName>": {...}}, "turns": [{"user": "<what the user says>", "reference": ["func(arg=value)", ...]}, ...]}

Rules:
- initial_config: the starting state for each stateful tool set, in exactly the form its _load_scenario reads
  (MathAPI has no state: leave it out). Make it specific and realistic: real-looking names, files with contents,
  balances, bookings, messages. The file system's root holds one top folder for the user; keep it to at most 2 layers.
- turns: exactly as many as the plan below, each needing exactly the number of calls the plan gives it. Each turn is one
  message from a user who does not know the function names. Later turns may build on earlier ones and on what the tools
  returned. Most turns ask for one thing.
- reference: the SHORTEST set of calls that does exactly what the user asked, in order, in Python call syntax with keyword
  arguments, using only the listed functions. Values must follow from what the user said, the starting state, or earlier
  results. It is scored by comparing both the final state and every result the reference's calls returned, so:
  - no extra checks, confirmations or lookups. A read-only call (a search, a list, a status or info check, a lookup
    of an id or symbol) belongs in the reference only if the user asks to see that information, or if the task cannot
    be done without a value only that call gives (the user did not give it and it is not in an earlier result).
  - no extra changes: do nothing the user did not ask for.
- Every request must give all the information the calls need, and have ONE clearly right outcome. Spell out every value
  that decides the outcome: exact texts, names, amounts, paths, and which value goes in which field (for example
  hashtags as tags or inside the post's text). A careful assistant reading the request should see only one reasonable
  way to do it.
- Respect the tools' real behaviour as the source shows it (for example logging in first where it is required, or being in
  the right folder). The reference must run without any error.
- No request may depend on something only a tool's answer reveals ("if the tire pressure is low, find a shop"): say
  exactly what to do.
- Do not mention function names or the reference in the user's words.
"""


REVIEW = """Review this task against the rules, as a strict editor. For each turn check: (1) the reference has no call the
request does not need (an extra lookup, check, confirmation or change); (2) the request states every value the outcome
depends on, so only one reasonable way to do it remains; (3) a careful assistant could do it from the request and earlier
results alone; (4) no request depends on something only a tool's answer reveals; (5) each turn still needs exactly the
number of calls the plan gives it. Fix every problem by editing the request or the reference (keep the starting state unless it must change).
Return the whole task as JSON in the same form, changed or not."""


# v4 ("implicit"): requests state the goal and the values that decide it, but leave out the steps a competent assistant
# works out from the tools and the state, as BFCL's own questions do. Still one right outcome; the solver check stays.
IMPLICIT = """
Style: GOAL, NOT PROCEDURE. Write each request the way a busy user would: say what they want done and every value that
decides the result, but leave out the steps a competent assistant can work out from the tools and the current state:
- moving into the right folder before acting on a file there, or creating a folder the user names as the destination;
- looking up an id or symbol for something the user names (a person, a company, a city, an airport);
- logging in with credentials the user gave earlier, or a prerequisite the tools require (for example pressing the brake
  pedal before starting the engine, or authenticating before booking);
- reusing a value an earlier turn produced (an order id, a booking id, a token).
Never leave out a value the end result depends on (texts, amounts, names, which field gets which value): the outcome must
stay unambiguous; only the route to it is left to the assistant. The reference still lists every call, the implied steps
included, and turn budgets in the plan count those calls too."""
IMPLICIT_REVIEW = (" Style check: each request states the goal and its deciding values, not the steps; rewrite requests that"
                   " spell out obvious steps (changing folder, looking up an id, logging in again, a required prerequisite),"
                   " but keep every value the result depends on.")


# v5 ("bfcl"): BFCL's own construction recipe, as its multi-turn write-up gives it, on top of v4's style: each task is
# built around an execution path sampled from the function dependency graph (lab/e13/graph.py), the user is a persona
# from PersonaHub (BFCL used the same source), and the starting state is hidden and must be explored.
BFCL = """
How this task is built (BFCL's method):
- EXECUTION PATH: the task is built around the path of functions given below. The reference follows it in order; add
  only the calls the tools need on the way (a prerequisite the simulator enforces, a lookup of a value the user cannot
  know), and drop a step only if it cannot fit the turn plan.
- PERSONA: the user is the person given below. Write every request in their own voice, about their own concerns. They
  talk about goals, not tools, and may ask for something no single function does (it then takes several calls).
- HIDDEN STATE: the assistant does not see the starting state and must find its way with the tools. Make it rich and
  lived-in, with more than the task needs: a file system up to 3 layers deep (this replaces the 2-layer limit above),
  where the files the user names sit in folders the user may name without saying where they are; a car as its owner
  left it; accounts, orders, bookings, messages, tickets with history.
- The ground truth is minimal and still covers every implicit step, as BFCL's does: what the simulator requires before
  an action (being in the right folder, pressing the brake before starting the engine, logging in) and any value that
  must be looked up first. Some turns may simply ask for information that only the tools hold."""
BFCL_REVIEW = (" Also check the BFCL method: the reference follows the given path (plus only the steps the tools require),"
               " the requests speak in the persona's voice about goals, and every value that decides the result is still stated.")


def tool_text(classes: list[str]) -> str:
    out = []
    for c in classes:
        doc, src = CLASSES[c]
        funcs = [json.loads(l) for l in open(PKG / "data/multi_turn_func_doc" / f"{doc}.json") if l.strip()]
        out.append(f"## Tool set {c}\n\n### Function descriptions\n{json.dumps(funcs, indent=1)}\n\n"
                   f"### Simulator source ({src}.py)\n```python\n{(SRC / f'{src}.py').read_text()}\n```")
    return "\n\n".join(out)


def run_reference(task: dict, classes: list[str], tag: str) -> tuple[str | None, list[str]]:
    """Runs the reference turn by turn in the simulators: (None if it ran cleanly, else what went wrong; the calls run so
    far with what each returned, for the writer to fix a failed task from, e.g. a login token the simulator issued)."""
    seen = []
    for t, turn in enumerate(task["turns"]):
        calls = turn.get("reference") or []
        if not calls or not str(turn.get("user", "")).strip():
            return f"turn {t + 1} has no user message or no calls", seen
        try:
            results, _ = execute_multi_turn_func_call(calls, task.get("initial_config", {}), classes, tag, "practice")
        except Exception as e:  # malformed call strings
            return f"turn {t + 1}: {e}", seen
        for call, res in zip(calls, results):
            seen.append(f"turn {t + 1}: {call} -> {res[:400]}")
            if res.startswith("Error during execution") or '"error"' in res[:200]:
                return f"turn {t + 1}, call {call}: {res[:300]}", seen
    return None, seen


def path_of(task: dict, classes: list[str]) -> list[str]:
    owner = {}
    for c in classes:
        for l in open(PKG / "data/multi_turn_func_doc" / f"{CLASSES[c][0]}.json"):
            if l.strip():
                owner[json.loads(l)["name"]] = c
    names = [re.match(r"\s*([\w.]+)\s*\(", call).group(1) for turn in task["turns"] for call in turn["reference"]]
    return [f"{owner.get(n, '?')}.{n}" for n in names]


GTYPE = {"dict": "object", "object": "object", "float": "number", "number": "number", "integer": "integer", "int": "integer",
         "string": "string", "boolean": "boolean", "array": "array", "tuple": "array", "list": "array"}


def _schema(p: dict) -> dict:
    """BFCL's parameter schema in the subset Gemini's function declarations accept."""
    t = GTYPE.get(p.get("type"), "string")
    out = {"type": t, **({"description": p["description"]} if p.get("description") else {})}
    if p.get("enum"):
        out["enum"] = [str(e) for e in p["enum"]]
    if t == "object" and p.get("properties"):
        out["properties"] = {k: _schema(v) for k, v in p["properties"].items()}
        req = [r for r in p.get("required", []) if r in p["properties"]]
        if req:
            out["required"] = req
    if t == "array":
        out["items"] = _schema(p.get("items") or {"type": "string"})
    return out


def functions(classes: list[str]) -> list[dict]:
    return [json.loads(l) for c in classes for l in open(PKG / "data/multi_turn_func_doc" / f"{CLASSES[c][0]}.json") if l.strip()]


def declarations(funcs: list[dict]) -> list[dict]:
    out = []
    for f in funcs:
        d = {"name": f["name"], "description": f.get("description", "")[:1000]}
        if (f.get("parameters") or {}).get("properties"):
            d["parameters"] = _schema(f["parameters"])
        out.append(d)
    return out


def call_string(name: str, args: dict, funcs: dict) -> str:
    props = ((funcs.get(name) or {}).get("parameters") or {}).get("properties", {})
    fix = lambda k, v: int(v) if isinstance(v, float) and v.is_integer() and props.get(k, {}).get("type") in ("integer", "int") else v
    return f"{name}({', '.join(f'{k}={fix(k, v)!r}' for k, v in (args or {}).items())})"


def retry(fn, tries: int = 6):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(min(60, 5 * 2 ** k))


SOLVER = ("You are an assistant that carries out the user's requests with the tools provided. Use the tools to do exactly "
          "what is asked, then answer briefly. Do not ask the user questions.")


def solve(llm, task: dict, classes: list[str], tag: str, max_steps: int = 8) -> tuple[bool, str]:
    """A solver works the task through the simulators, turn by turn, without the reference; then BFCL's own multi-turn
    checker scores it against the reference. (passed, what went wrong)."""
    funcs = functions(classes)
    by_name, tools = {f["name"]: f for f in funcs}, [{"functionDeclarations": declarations(funcs)}]
    contents, decoded = [], []
    for t, turn in enumerate(task["turns"]):
        contents.append({"role": "user", "parts": [{"text": turn["user"]}]})
        steps = []
        for _ in range(max_steps):
            reply = retry(lambda: llm.content({"systemInstruction": {"parts": [{"text": SOLVER}]}, "contents": contents,
                                               "tools": tools, "generationConfig": {"temperature": 1.0,
                                               "thinkingConfig": {"thinkingLevel": "medium"}}}))
            reply.setdefault("role", "model")
            contents.append(reply)
            fcs = [p["functionCall"] for p in reply.get("parts", []) if "functionCall" in p]
            if not fcs:
                break
            calls = [call_string(fc["name"], fc.get("args"), by_name) for fc in fcs]
            steps.append(calls)
            results, _ = execute_multi_turn_func_call(calls, task.get("initial_config", {}), classes, f"solver{tag}", "practice")
            contents.append({"role": "user", "parts": [{"functionResponse": {"name": fc["name"], "response": {"result": r}}}
                                                       for fc, r in zip(fcs, results)]})
        decoded.append(steps)
    entry = {"id": f"multi_turn_base_{tag}", "initial_config": task.get("initial_config", {}), "involved_classes": classes}
    try:
        res = multi_turn_checker(decoded, [turn["reference"] for turn in task["turns"]], entry, "multi_turn_base", f"check{tag}")
    except Exception as e:
        return False, f"checker error: {e}"[:300]
    if res.get("valid"):
        return True, ""
    return False, (f"{res.get('error_message')} The solver's calls per turn: {json.dumps(decoded)[:1500]}. "
                   f"Details: {json.dumps(res.get('details', {}), default=str)[:1200]}")


def shape(task: dict, plan: list[int]) -> str | None:
    """None if the task follows its plan: that many turns, each with at most its planned number of calls."""
    if len(task["turns"]) != len(plan):
        return f"the task has {len(task['turns'])} turns; the plan asks for {len(plan)}"
    over = [f"turn {k + 1} has {len(t.get('reference') or [])} calls (plan: {n})"
            for k, (t, n) in enumerate(zip(task["turns"], plan)) if len(t.get("reference") or []) > n]
    return "; ".join(over) or None


def walk(rng: random.Random, edges: list[dict], classes: list[str], length: int) -> list[str]:
    """An execution path through the dependency graph, inside the task's tool sets, starting in the main set: a random
    walk along edges, jumping to a fresh function when it runs out (a function is not repeated)."""
    nodes = [f"{c}.{f['name']}" for c in classes for f in functions([c])]
    out_of: dict[str, list[str]] = {}
    for e in edges:
        if e["from"] in nodes and e["to"] in nodes:
            out_of.setdefault(e["from"], []).append(e["to"])
    start = [n for n in nodes if n.startswith(classes[0] + ".") and n in out_of] or nodes
    path = [rng.choice(start)]
    while len(path) < length:
        nxt = [n for n in out_of.get(path[-1], []) if n not in path]
        if not nxt:  # dead end: continue from any unused function that leads somewhere
            nxt = [n for n in out_of if n not in path] or [n for n in nodes if n not in path]
            if not nxt:
                break
        path.append(rng.choice(nxt))
    return path


class Writer:
    def __init__(self, model: str, seed: int, style: str = "explicit"):
        self.llm, self.rng, self.lock, self.style = Gemini(model), random.Random(seed), threading.Lock(), style
        self.system = SYSTEM + (IMPLICIT if style in ("implicit", "bfcl") else "") + (BFCL if style == "bfcl" else "")
        self.review = (REVIEW + (IMPLICIT_REVIEW if style in ("implicit", "bfcl") else "")
                       + (BFCL_REVIEW if style == "bfcl" else ""))
        if style == "bfcl":
            self.edges = json.load(open(HERE / "graph.json"))["edges"]
            self.personas = [json.loads(l)["persona"] for l in open(HERE / "personas.jsonl") if l.strip()]
        self.recent: dict[tuple, list[str]] = {}

    def draw(self) -> tuple[list[str], str, list[int], list[str]]:
        """Tool sets, the user (a theme, or a persona in v5), calls per turn, and in v5 the execution path."""
        with self.lock:
            n = self.rng.choices(list(TURNS), weights=list(TURNS.values()))[0]
            plan = self.rng.choices(list(CALLS), weights=list(CALLS.values()), k=n)
            if self.style == "bfcl":
                classes = [self.rng.choice(PRIMARY)]
                if self.rng.random() >= ALONE:
                    classes.append(self.rng.choices(list(COMPANION), weights=list(COMPANION.values()))[0])
                return classes, self.rng.choice(self.personas), plan, walk(self.rng, self.edges, classes, sum(plan))
            names = list(CLASSES)
            classes = [self.rng.choice(names)] if self.rng.random() < 1 / 3 else self.rng.sample(names, 2)
            return classes, self.rng.choice(THEMES), plan, []

    def write(self, i: int, retries: int = 2) -> tuple[dict | None, dict]:
        classes, theme, plan, route = self.draw()
        avoid = self.recent.get(tuple(sorted(classes)), [])[-12:]
        user = (f"{tool_text(classes)}\n\nTool sets in this task: {', '.join(classes)}"
                + (" (use both)" if len(classes) > 1 else "")
                + (f".\nThe user (persona): {theme}\n" if route else f".\nThe user: {theme}.\n")
                + (f"Execution path: {' -> '.join(route)}\n" if route else "")
                + f"Plan: {len(plan)} turns; calls needed per turn, in order: {', '.join(map(str, plan))}.\n"
                + ("Already written for these tools, so do something different:\n- " + "\n- ".join(avoid) + "\n" if avoid else "")
                + "Write the task.")
        log = {"i": i, "classes": classes, "theme": theme, "plan": plan, **({"route": route} if route else {})}
        msgs, reviewed, solver_fails = user, False, 0
        for attempt in range(retries + 4):  # repairs, one review, one fix after a failed solve
            seen = []
            try:
                task = json.loads(retry(lambda: self.llm.generate(self.system, msgs, json_mode=True, temperature=1.0,
                                                                  max_tokens=16000, thinking="high", timeout=600), 3))
                if isinstance(task, list) and len(task) == 1:
                    task = task[0]
                if isinstance(task, dict) and "turns" not in task:  # e.g. {"task": {...}} after a review
                    task = next((v for v in task.values() if isinstance(v, dict) and "turns" in v), task)
                bad = None if isinstance(task, dict) and isinstance(task.get("turns"), list) else "not the JSON asked for"
            except Exception as e:
                task, bad = None, f"writer error: {e}"[:300]
            if task is not None and bad is None:
                task["initial_config"] = {k: v for k, v in (task.get("initial_config") or {}).items() if k in classes}
                bad, seen = run_reference(task, classes, f"w{i}a{attempt}")
                bad = bad or shape(task, plan)
            if bad is None and not reviewed:
                # it runs: one strict review against the rules (no needless calls, no ambiguity), then run it again
                reviewed = True
                log["reviewed"] = True
                msgs = user + "\n\nYour task:\n" + json.dumps(task) + "\n\n" + self.review
                continue
            if bad is None:
                ok, why = solve(self.llm, task, classes, f"{i}a{attempt}")
                if ok:
                    with self.lock:
                        self.recent.setdefault(tuple(sorted(classes)), []).append(task["turns"][0]["user"][:160])
                    return task | {"classes": classes, "theme": theme, "plan": plan, **({"route": route} if route else {})}, log | {"attempts": attempt + 1}
                solver_fails += 1
                log.setdefault("problems", []).append(f"solver: {why[:400]}")
                if solver_fails > 1:
                    return None, log
                msgs = (user + "\n\nYour task:\n" + json.dumps(task) + "\n\nA careful assistant who did not see the "
                        f"reference tried it and was scored wrong: {why}\nIf the request was unclear, left room for another "
                        "reasonable way, or the reference has a call the request does not need, fix the request or the "
                        "reference so only one reasonable way remains. Keep the plan. Return the whole corrected task.")
                continue
            log.setdefault("problems", []).append(bad)
            if task is None:
                continue
            msgs = (user + "\n\nYour task:\n" + json.dumps(task) + f"\n\nIt failed a check: {bad}\n"
                    + ("What the calls returned, up to the failure:\n" + "\n".join(seen) + "\n" if seen else "")
                    + "Fix the task (the starting state, the requests or the reference) and return the whole corrected task.")
        return None, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--status", default=None)
    ap.add_argument("--style", choices=["explicit", "implicit", "bfcl"], default="explicit",
                    help="explicit: every step spelled out (v3); implicit: goal and deciding values only, as BFCL (v4); "
                         "bfcl: v4 plus BFCL's recipe: main/companion sets, graph paths, personas, hidden state (v5)")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "possible_answer").mkdir(parents=True, exist_ok=True)
    writer, kept, rejected, started = Writer(a.model, a.seed, a.style), [], [], time.time()
    # every task is saved as it is kept or dropped, so a crash or a restart resumes instead of starting over
    kept_log, rej_log = out / "kept.jsonl", out / "rejected.jsonl"
    if kept_log.exists():
        kept = [json.loads(l) for l in kept_log.open() if l.strip()][:a.n]
    if rej_log.exists():
        rejected = [json.loads(l) for l in rej_log.open() if l.strip()]

    def status(stage):
        if a.status:
            Path(a.status).write_text(json.dumps({"stage": stage, "kept": len(kept), "rejected": len(rejected), "target": a.n,
                                                  "model": a.model, "started": started, "updated": time.time()}))

    # ask for some extra, since a share fails its own reference; stop once n are kept
    todo = max(0, int((a.n - len(kept)) * 1.4))
    with ThreadPoolExecutor(a.workers) as pool, kept_log.open("a") as kf, rej_log.open("a") as rf:
        futures = [pool.submit(writer.write, len(kept) + len(rejected) + i) for i in range(todo)]
        for f in as_completed(futures):
            if f.cancelled():
                continue
            try:
                task, log = f.result()
            except Exception as e:  # one task's failure never stops the run
                task, log = None, {"error": repr(e)[:300]}
            if task is None:
                rejected.append(log)
                rf.write(json.dumps(log, default=str) + "\n"); rf.flush()
            elif len(kept) < a.n:
                kept.append(task)
                kf.write(json.dumps(task) + "\n"); kf.flush()
            if len(kept) >= a.n:
                for g in futures:
                    g.cancel()
            if (len(kept) + len(rejected)) % 10 == 0:
                status("writing")

    with open(out / "BFCL_v4_multi_turn_base.json", "w") as q, open(out / "possible_answer/BFCL_v4_multi_turn_base.json", "w") as g, \
            open(out / "tasks.jsonl", "w") as t:
        for i, task in enumerate(kept):
            tid = f"multi_turn_base_{i}"
            q.write(json.dumps({"id": tid, "question": [[{"role": "user", "content": turn["user"]}] for turn in task["turns"]],
                                "initial_config": task["initial_config"], "path": path_of(task, task["classes"]),
                                "involved_classes": task["classes"], "excluded_function": []}) + "\n")
            g.write(json.dumps({"id": tid, "ground_truth": [turn["reference"] for turn in task["turns"]]}) + "\n")
            t.write(json.dumps({"id": tid, **task}) + "\n")
    status("done")
    print(f"kept {len(kept)}, rejected {len(rejected)}")


if __name__ == "__main__":
    main()

"""τ²-bench as a track for a life (E10): run tasks with the creature as the agent, read the outcome, and turn the
conversations the creature should learn from into training examples.

τ²-bench runs in its own environment (~/tau2-bench, its own venv), so it is driven through its command line.
The creature is reached through a small proxy in front of vLLM that records every request the agent makes
(messages and tools) and the reply; a lesson is those exact turns, rendered by the chat template the way vLLM
rendered them, so the creature learns what it actually saw.
"""

import json
import os
import random
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

TAU2 = Path(os.environ.get("TAU2_DIR", Path.home() / "tau2-bench"))
USER_LLM = os.environ.get("TAU2_USER_LLM", "vertex_ai/gemini-3-flash-preview")
END = re.compile(r"###[\w-]+###")


def split(domain: str, name: str) -> list[str]:
    """Task ids of an official split (train / test) of one domain."""
    return json.loads((TAU2 / "data" / "tau2" / "domains" / domain / "split_tasks.json").read_text())[name]


def day_tasks(domains: list[str], n: int, seed: int, day: int) -> dict[str, list[str]]:
    """n training tasks for one day, drawn from the domains' training splits together (so each domain appears in
    proportion to its size), different each day; a task may come back on a later day."""
    pool = [(d, t) for d in domains for t in split(d, "train")]
    picked = random.Random(seed * 100003 + day).sample(pool, min(n, len(pool)))
    out: dict[str, list[str]] = {}
    for d, t in picked:
        out.setdefault(d, []).append(t)
    return out


def run(domain: str, task_ids: list[str] | None, split_name: str | None, agent: str, api_base: str, save_to: Path,
        seed: int, agent_temperature: float, user_temperature: float, concurrency: int = 6) -> list[dict]:
    """Run τ²-bench tasks and return its simulations. task_ids picks tasks; otherwise split_name (e.g. "test")."""
    save_to.parent.mkdir(parents=True, exist_ok=True)
    if save_to.exists():
        import shutil

        shutil.rmtree(save_to)
    args = {"api_base": api_base, "temperature": agent_temperature, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    cmd = [str(TAU2 / ".venv" / "bin" / "tau2"), "run", "--domain", domain, "--num-trials", "1", "--seed", str(seed),
           "--max-concurrency", str(concurrency), "--agent-llm", f"hosted_vllm/{agent}", "--agent-llm-args", json.dumps(args),
           "--user-llm", USER_LLM, "--user-llm-args", json.dumps({"temperature": user_temperature}), "--save-to", str(save_to.resolve())]
    cmd += ["--task-ids", *task_ids] if task_ids else ["--task-split-name", split_name]
    log = save_to.parent / f"{save_to.name}.log"
    with log.open("w") as f:
        subprocess.run(cmd, cwd=TAU2, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False,
                       env={**os.environ, "VERTEXAI_LOCATION": "global"})
    results = save_to / "results.json"
    if not results.exists():
        raise RuntimeError(f"τ²-bench wrote no results; see {log}")
    return json.loads(results.read_text())["simulations"]


def reward(sim: dict) -> float | None:
    """The task's true outcome (1 done, 0 not), or None when the run failed for infrastructure reasons."""
    return (sim.get("reward_info") or {}).get("reward")


def ending(sim: dict) -> dict:
    """What the sense reads: the customer's opening request, their last real words, and how the conversation ended."""
    users = [(m.get("content") or "").strip() for m in sim["messages"] if m["role"] == "user"]
    said = [u for u in (END.sub("", u).strip() for u in users) if u]
    ended = any(END.search(u) for u in users)
    return {"request": said[0][:500] if said else "", "said": said[-1][:500] if len(said) > 1 else "",
            "next_move": "left the session" if ended else "replied"}


def first_user(messages: list[dict]) -> str:
    return next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")


class Recorder:
    """An HTTP proxy in front of vLLM's OpenAI server that appends each chat completion (request and reply) to a
    file, so a day's conversations can be turned into lessons exactly as the agent saw them."""

    def __init__(self, upstream: str, port: int):
        self.upstream, self.port, self.path, self.lock = upstream.rstrip("/"), port, None, threading.Lock()
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _forward(self, method):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
                r = requests.request(method, recorder.upstream + self.path, data=body, timeout=600,
                                     headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
                if method == "POST" and self.path.endswith("/chat/completions") and r.ok and recorder.path:
                    with recorder.lock, open(recorder.path, "a") as f:
                        f.write(json.dumps({"request": json.loads(body), "response": r.json()}, ensure_ascii=False) + "\n")
                self.send_response(r.status_code)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(r.content)))
                self.end_headers()
                self.wfile.write(r.content)

            def do_POST(self):
                self._forward("POST")

            def do_GET(self):
                self._forward("GET")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def record_to(self, path: Path | None):
        self.path = path


def lessons(records_file: Path, sims: list[dict], tokenizer) -> dict[str, list[dict]]:
    """Training examples per simulation id: every agent turn of that conversation, rendered as the chat template
    rendered it for vLLM (system policy, tools, history), with the agent's reply as the target."""
    by_opening: dict[str, list[dict]] = {}
    for line in records_file.open() if records_file.exists() else []:
        rec = json.loads(line)
        by_opening.setdefault(first_user(rec["request"]["messages"]), []).append(rec)
    out = {}
    for sim in sims:
        opening = first_user([m for m in sim["messages"]])
        recs = by_opening.get(opening, [])
        rows = []
        for rec in recs:
            req, msg = rec["request"], rec["response"]["choices"][0]["message"]
            reply = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                reply["tool_calls"] = msg["tool_calls"]
            kw = {"tools": req.get("tools"), "tokenize": False, "enable_thinking": False}
            prompt = tokenizer.apply_chat_template(req["messages"], add_generation_prompt=True, **kw)
            full = tokenizer.apply_chat_template(req["messages"] + [reply], **kw)
            if not full.startswith(prompt):
                continue  # the template rendered the reply differently from a generation; skip rather than mislearn
            rows.append({"prompt_text": prompt, "content": full[len(prompt):].rstrip("\n")})
        if rows:
            out[sim["id"]] = rows
    return out

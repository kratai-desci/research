"""The shared agent loop: think → act → observe, identical for every system.

The model answers in forced JSON: {"thought", "action", "input"}. Actions are the environment's tools plus
"answer", which ends the episode. Token accounting:
  input    = the prompt the model reads, every step
  thinking = the `thought` text plus tool-call inputs (the skill signal)
  output   = the final answer text
"""

import json
from dataclasses import dataclass, field
from typing import Callable

from lab.models import OllamaModel, count_tokens

SYSTEM_PROMPT = """You are a personal assistant for one user. You talk with them and can look things up.

Each turn, reply with JSON only:
{"thought": "<brief reasoning>", "action": "<one of: ACTIONS>", "input": "<text>"}

TOOLS
Use "answer" with your final reply to the user as input. It ends your turn."""

MAX_STEPS = 6


def parse_json(text: str) -> dict | None:
    """The first JSON object in the reply (models without forced JSON sometimes add text around it)."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                try:
                    data = json.loads(text[start : i + 1])
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


@dataclass
class Step:
    n: int
    thought: str
    action: str
    input: str
    observation: str
    input_tokens: int
    thinking_tokens: int
    output_tokens: int
    latency_ms: int


@dataclass
class Episode:
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    # For sleep: the exact messages the model saw before each step, and its reply as clean JSON
    prompts: list[list[dict]] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action != "answer")

    def total(self, attr: str) -> int:
        return sum(getattr(s, attr) for s in self.steps)


def run_episode(
    model: OllamaModel,
    tools: dict[str, tuple[str, Callable[[str], str]]],
    history: list[dict],
    message: str,
    on_step: Callable[[Step], None] | None = None,
    max_steps: int = MAX_STEPS,
    is_done: Callable[[], bool] | None = None,
    allow_answer: bool = True,
    keep_observations: int | None = None,
) -> Episode:
    """tools: name -> (description, function). history: earlier turns with the user, if any.
    For environments that end themselves (e.g. WebShop's Buy Now), pass is_done and allow_answer=False."""
    actions = [*tools, "answer"] if allow_answer else list(tools)
    tool_text = "\n".join(f'- "{name}": {desc}' for name, (desc, _) in tools.items()) or "- (no tools)"
    system = SYSTEM_PROMPT.replace("ACTIONS", ", ".join(actions)).replace("TOOLS", "Tools:\n" + tool_text)
    if not allow_answer:
        system = system.replace('Use "answer" with your final reply to the user as input. It ends your turn.', "The task ends when the environment says so.")
    schema = {
        "type": "object",
        "properties": {"thought": {"type": "string"}, "action": {"enum": actions}, "input": {"type": "string"}},
        "required": ["thought", "action", "input"],
    }
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": message}]
    episode = Episode()

    for n in range(1, max_steps + 1):
        reply = model.chat(messages, schema)
        data = parse_json(reply.content)
        if data is not None:
            thought, action, text = str(data.get("thought", "")), str(data.get("action", "answer")), str(data.get("input", ""))
        else:
            thought, action, text = "", "answer", reply.content
        is_answer = allow_answer and (action == "answer" or action not in tools)
        if not allow_answer and action not in tools:
            action = actions[0]
        observation = "" if is_answer else tools[action][1](text)
        step = Step(
            n=n,
            thought=thought,
            action="answer" if is_answer else action,
            input=text,
            observation=observation,
            input_tokens=reply.input_tokens,
            thinking_tokens=count_tokens(thought) + (0 if is_answer else count_tokens(text)),
            output_tokens=count_tokens(text) if is_answer else 0,
            latency_ms=reply.latency_ms,
        )
        episode.steps.append(step)
        episode.prompts.append(list(messages))
        episode.replies.append(json.dumps({"thought": thought, "action": step.action, "input": text}, ensure_ascii=False))
        if on_step:
            on_step(step)
        if is_answer:
            episode.answer = text
            break
        if is_done and is_done():
            break
        messages += [
            {"role": "assistant", "content": episode.replies[-1]},
            {"role": "user", "content": f"Observation from {action}:\n{observation}"},
        ]
        if keep_observations:  # keep only the last few pages in full; older ones shrink to a stub
            obs_idx = [i for i, m in enumerate(messages) if m["role"] == "user" and m["content"].startswith("Observation from ")]
            for i in obs_idx[:-keep_observations]:
                messages[i] = {"role": "user", "content": messages[i]["content"].split("\n", 1)[0] + "\n(earlier page omitted)"}
    return episode

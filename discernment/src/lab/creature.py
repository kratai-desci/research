"""The creature's new faculties: reflecting on an attempt, retrying after feedback, and turning a
corrected retry into learning material (corrected examples and good/bad preference pairs).

The creature never sees the true reward. Its learning signal comes from the user's reaction, its own
reflection, or both, depending on the life's `signal` setting.
"""

import json
from dataclasses import dataclass

from lab.agent import Episode, parse_json

REFLECT_SYSTEM = """You review your own past attempt at a shopping task, honestly.
Reply with JSON only: {"verdict": "good" or "bad", "lesson": "<one short sentence for next time>"}
"good" means the purchase matches the request. Say "bad" if anything clearly does not match."""

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"enum": ["good", "bad"]}, "lesson": {"type": "string"}},
    "required": ["verdict", "lesson"],
}

RETRY_NOTE = 'Your last attempt at this task did not go well. The user said: "{reaction}". Your lesson: {lesson}'


@dataclass
class Verdict:
    good: bool | None  # None = no usable signal (e.g. the user was silent and the signal is "user")
    lesson: str
    source: str  # oracle | user | self | both | sense | none
    weight: float = 1.0  # how much a good attempt counts in sleep (the sense's "acceptable" counts half)


def summarize(episode: Episode, limit: int = 8) -> str:
    """What the creature did, step by step, for reflection."""
    lines = [f"{s.n}. {s.action}[{s.input}]" for s in episode.steps[:limit]]
    if len(episode.steps) > limit:
        lines.append(f"... ({len(episode.steps) - limit} more steps)")
    return "\n".join(lines)


def reflect(model, instruction: str, episode: Episode, bought: str, reaction: str | None) -> tuple[bool | None, str]:
    """The creature looks back at its own attempt. Same model, deliberate pass."""
    said = f'The user said: "{reaction}"' if reaction else "The user said nothing."
    user = (
        f"Task: {instruction}\n\nWhat you did:\n{summarize(episode)}\n\n"
        f"Result: {bought or 'nothing was bought'}\n{said}\n\nWas your attempt good?"
    )
    reply = model.chat([{"role": "system", "content": REFLECT_SYSTEM}, {"role": "user", "content": user}], REFLECT_SCHEMA)
    data = parse_json(reply.content) or {}
    verdict = data.get("verdict")
    return (None if verdict not in ("good", "bad") else verdict == "good"), str(data.get("lesson", ""))


SENSE_LESSON = {"20": "the user wants an adjustment; change only what they asked for",
                "0": "the attempt was wrong; start again and read the request carefully"}


def judge(signal: str, model, instruction: str, episode: Episode, bought: str, reaction, reward: float, threshold: float,
          sense=None) -> Verdict:
    """The learning signal for one attempt, per the life's signal setting. `sense` (a SenseReader) scores the
    user's reaction for the sense signals: 100 and 80 are good (80 at half weight), 20 and 0 bad, null no signal."""
    if signal in ("sense", "sense-coupled"):
        move = reaction.next_move or ("replied" if reaction.text else "said nothing")
        label, conf = sense.score({"request": instruction, "next_move": move, "said": reaction.text})
        if label == "null":
            return Verdict(None, f"unclear ({conf:.2f})", signal)
        return Verdict(label in ("80", "100"), SENSE_LESSON.get(label, f"scored {label} ({conf:.2f})"), signal,
                       weight=0.5 if label == "80" else 1.0)
    if signal == "oracle":
        return Verdict(reward >= threshold, "", "oracle")
    if signal == "user":
        if reaction.kind in ("silent", "settle", "move_on"):
            return Verdict(None, "", "user")
        return Verdict(reaction.kind in ("perfect", "praise"), "", "user")
    if signal == "self":
        good, lesson = reflect(model, instruction, episode, bought, None)
        return Verdict(good, lesson, "self")
    if signal == "both":
        heard = reaction.text or ("(no comment; they moved on to their next request)" if reaction.next_move == "new request" else None)
        good, lesson = reflect(model, instruction, episode, bought, heard)
        return Verdict(good, lesson, "both")
    return Verdict(None, "", "none")


def examples(episode: Episode, strip_note: str = "", weight: float = 1.0, once: bool = False) -> list[list[dict]]:
    """Each step as a training example: the prompt it saw, then its reply. `strip_note` removes the retry
    note from the first message, so the creature learns to act correctly without being told. A weight other than 1
    rides on the reply message, where the torch sleep reads it. `once` (E8) splits the weight across the steps, so a
    lesson counts once however long it was: a 15-step failure no longer outweighs a 3-step purchase five to one."""
    if once and episode.replies:
        weight /= len(episode.replies)
    out = []
    for prompt, reply in zip(episode.prompts, episode.replies):
        prompt = [dict(m) for m in prompt]
        if strip_note:
            for m in prompt:
                if m["role"] == "user" and strip_note in m["content"]:
                    m["content"] = m["content"].replace(f"\n\n{strip_note}", "")
        out.append(prompt + [{"role": "assistant", "content": reply, **({"weight": weight} if weight != 1.0 else {})}])
    return out


def preference_pair(failed: Episode, corrected: Episode, note: str) -> dict | None:
    """Where the corrected retry first acted differently from the failed attempt (same prompt, with the
    retry note removed), make a pair: prefer the corrected reply over the failed one."""
    fixed = examples(corrected, strip_note=note)
    for i, (bad, good) in enumerate(zip(examples(failed), fixed)):
        if bad[:-1] != good[:-1]:
            return None  # prompts diverged before any differing action
        if json.loads(bad[-1]["content"]) != json.loads(good[-1]["content"]):
            return {"prompt": good[:-1], "chosen": good[-1]["content"], "rejected": bad[-1]["content"]}
    return None

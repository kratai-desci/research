"""The feedback sense: a small frozen model that reads how a user reacted and scores the attempt.

Scores are two clear groups with a gap between them, so the creature only learns from clear signals:
  100 exactly right · 80 acceptable  |  null unclear  |  20 needs adjusting · 0 wrong

It reads the request and the user's next move (what kind of turn it was and the words), not the AI's result,
so it stays a reader of people rather than a hidden judge of the work (prompt(with_result=True) gives the
other variant).

Modules: data.py builds the datasets (real WildFB reactions; shopping held out as the unseen test); train.py
fine-tunes a LoRA on Qwen3-0.6B and evaluates it; status.py reports progress for the dashboard.
"""

LABELS = ["0", "20", "80", "100", "null"]  # first tokens 0, 2, 8, 1, n are distinct, so one token decides

MEANING = {
    "100": "exactly right: the user praises it explicitly",
    "80": "acceptable: the user moves on to a new, different request without complaint, or says a plain 'ok, thanks'",
    "20": "needs adjusting: the user asks to adjust the same result (a different size, colour, length, tone, price...)",
    "0": "wrong: the user rejects it, repeats or rephrases the same request, gives up on it, or gets angry",
    "null": "unclear: the reaction does not show whether it worked: lukewarm ('I guess'), settling, leaving the "
            "session, an off-topic reply, 'hmm', or nothing at all",
}

NEXT_MOVES = ["replied", "new request", "asked to adjust", "repeated the request", "left the session", "said nothing"]

SYSTEM = ("You read how a user reacted to an AI assistant and score the assistant's attempt. Answer with one of: "
          "100 (exactly right), 80 (acceptable), 20 (needs adjusting), 0 (wrong), null (unclear).")


# v3 (guide3.md) asks two questions, each answered with one token: is the reaction a clear verdict on the result
# (yes / no; not clear means null), and if it is, which score (0, 20, 80, 100).
CLEAR = ["yes", "no"]
SCORES = ["0", "20", "80", "100"]
CLEAR_SYSTEM = ("You read how a user reacted to an AI assistant. Is the reaction a clear verdict on the assistant's result? "
                "Thanking the effort rather than the result, sarcasm, hedging, settling, silence, moving on without a word, "
                "or anything ambiguous is not clear. Answer yes or no.")
SCORE_SYSTEM = ("You read how a user reacted to an AI assistant; the reaction is a clear verdict on the result. Score the "
                "assistant's attempt: 100 (exactly right), 80 (acceptable, or building on it), 20 (one part missed what was "
                "asked), 0 (wrong).")


def prompt3(ex: dict, step: str) -> list[dict]:
    """The chat messages for one of v3's two questions: step "clear" or "score"."""
    lines = [f"Request: {ex['request']}", f"The user's next move: {ex['next_move']}",
             f"What the user said: {ex['said'] or '(nothing)'}", "Clear verdict:" if step == "clear" else "Score:"]
    return [{"role": "system", "content": CLEAR_SYSTEM if step == "clear" else SCORE_SYSTEM},
            {"role": "user", "content": "\n".join(lines)}]


def prompt(ex: dict, with_result: bool = False) -> list[dict]:
    """The chat messages the sense sees for one example (without the answer)."""
    lines = [f"Request: {ex['request']}"]
    if with_result:
        lines.append(f"What the assistant did: {ex['result']}")
    lines += [f"The user's next move: {ex['next_move']}", f"What the user said: {ex['said'] or '(nothing)'}", "Score:"]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]

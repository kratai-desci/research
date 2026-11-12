"""The simulated user: reacts in words to what the creature bought, by fixed rules from the hidden goal.

It never states a score. The rules are identical across lives so conditions stay comparable.
"""

import random
from dataclasses import dataclass

SILENCE_RATE = 0.2

PERFECT = ["Perfect, thanks!", "That's exactly it, thank you.", "Great, that's the one."]
WRONG_KIND = ["That's not what I asked for.", "No, that's a different kind of thing.", "Not what I wanted at all."]
NO_PURCHASE = ["Never mind.", "Forget it, I'll look myself.", "You didn't get me anything."]
CLOSE_OPTION = ["Close, but I wanted it in {x}.", "Almost, but I needed {x}.", "Not quite, I asked for {x}."]
CLOSE_ATTRIBUTE = ["Close, but I wanted something {x}.", "Almost, but it should be {x}.", "Not quite, I needed it {x}."]
OVER_BUDGET = ["Close, but that's over my budget.", "Nice, but too expensive for me.", "Almost, but it costs too much."]
NOT_QUITE = ["Hmm, not quite what I wanted.", "Not exactly what I had in mind.", "That's not quite right."]


@dataclass
class Reaction:
    kind: str  # perfect | close | wrong_kind | no_purchase | silent (the LLM user, lab/user_llm.py, has its own kinds)
    text: str  # what the user says ("" when silent)
    next_move: str = ""  # replied | new request | asked to adjust | ... ("" = replied, or said nothing when silent)
    persona: str = ""


def react(outcome: dict, rng: random.Random) -> Reaction:
    """outcome: the bridge's /outcome response for the finished task."""
    if rng.random() < SILENCE_RATE:
        return Reaction("silent", "")
    if not outcome["done"]:
        return Reaction("no_purchase", rng.choice(NO_PURCHASE))
    if outcome["reward"] >= 1.0:
        return Reaction("perfect", rng.choice(PERFECT))
    if outcome.get("info", {}).get("r_type", 1.0) < 0.5:
        return Reaction("wrong_kind", rng.choice(WRONG_KIND))

    goal, bought = outcome["goal"], outcome["purchased"]
    bought_options = {o.lower() for o in bought["options"]}
    missing = [("option", o) for o in goal["options"] if str(o).lower() not in bought_options]
    missing += [("attribute", a) for a in goal["attributes"] if a.lower() not in bought["text"]]
    if goal["price_upper"] and bought["price"] and bought["price"] > goal["price_upper"]:
        missing.append(("price", None))
    if not missing:
        return Reaction("close", rng.choice(NOT_QUITE))
    kind, value = rng.choice(missing)
    if kind == "price":
        return Reaction("close", rng.choice(OVER_BUDGET))
    template = rng.choice(CLOSE_OPTION if kind == "option" else CLOSE_ATTRIBUTE)
    return Reaction("close", template.format(x=value))

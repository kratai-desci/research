"""Track 1: WebShop, through the HTTP bridge in lab/scripts/webshop_server.py.

Goals are WebShop's human-written instructions, shuffled by WebShop with a fixed seed. Following the common
protocol, the first N_TEST goal indices are the held-out test set; the rest are the practice stream.
Score = WebShop's reward (0-1), stored as 0-100; success = reward 1.
"""

import os
import re

import requests

URLS = {
    "synthetic": os.environ.get("WEBSHOP_SYNTHETIC_URL", "http://127.0.0.1:3100"),
    "human": os.environ.get("WEBSHOP_HUMAN_URL", "http://127.0.0.1:3101"),
}
N_TEST = 100  # synthetic goals 0-99 = held-out probes; the human goals (13 on the small set) are all probes
MAX_STEPS = 15
MAX_CLICKABLES = 25
KEEP_OBSERVATIONS = 2  # pages kept in full in the context; older ones become stubs


def info(goal_set: str = "synthetic") -> dict:
    return requests.get(f"{URLS[goal_set]}/info", timeout=10).json()


def goal_indices(split: str, limit: int = 0, start: int = 0, goal_set: str = "synthetic") -> list[int]:
    total = info(goal_set)["goals"]
    if goal_set == "human":
        ids = list(range(total))
    else:
        ids = list(range(N_TEST)) if split == "test" else list(range(N_TEST, total))
    ids = ids[start:]
    return ids[:limit] if limit else ids


def render(obs: str, actions: dict) -> str:
    """WebShop's text observation, made readable, plus what can be clicked."""
    text = re.sub(r"\s*\[SEP\]\s*", "\n", obs).strip()
    clickables = [c for c in actions["clickables"] if c != "search"][:MAX_CLICKABLES]
    lines = [text]
    if clickables:
        lines.append("Clickable: " + " | ".join(clickables))
    if actions["search"]:
        lines.append("You can search.")
    return "\n".join(lines)


class WebShopSession:
    """One episode. `tools()` plugs into the agent loop; `done` ends it."""

    def __init__(self, goal: int, goal_set: str = "synthetic", note: str = ""):
        """note: extra context shown with the first page (e.g. the user's reaction to a previous attempt)."""
        self.url = URLS[goal_set]
        r = requests.post(f"{self.url}/reset", json={"session": goal}, timeout=60).json()
        self.instruction = r["instruction"]
        self.first_observation = render(r["observation"], r["actions"]) + (f"\n\n{note}" if note else "")
        self.actions = r["actions"]
        self.reward, self.done = 0.0, False

    def outcome(self) -> dict:
        """Hidden goal and purchase, for the simulated user and the observer only."""
        return requests.get(f"{self.url}/outcome", timeout=60).json()

    def _step(self, action: str) -> str:
        r = requests.post(f"{self.url}/step", json={"action": action}, timeout=60).json()
        self.reward, self.done, self.actions = float(r["reward"]), bool(r["done"]), r["actions"]
        return render(r["observation"], r["actions"])

    # WebShop silently ignores invalid actions; the harness says so instead (same for every system).
    def search(self, keywords: str) -> str:
        if not self.actions["search"]:
            return "Invalid: there is no search box on this page. Click 'back to search' first."
        return self._step(f"search[{keywords}]")

    def click(self, target: str) -> str:
        target = target.strip().lower()
        if target not in self.actions["clickables"]:
            options = " | ".join(c for c in self.actions["clickables"] if c != "search")[:600]
            return f"Invalid: '{target}' is not clickable on this page. Clickable: {options}"
        return self._step(f"click[{target}]")

    def tools(self) -> dict:
        return {
            "search": ("search the shop. Input: keywords.", self.search),
            "click": ("click a product, option or button. Input: its exact text from 'Clickable'. Clicking 'Buy Now' ends the task.", self.click),
        }


def checks(reward: float) -> list[tuple[str, str, bool, str]]:
    """(name, kind, passed, evidence) rows for the store."""
    return [
        ("Purchase made", "T", reward > 0, f"reward {reward:.2f}"),
        ("Success (reward = 1)", "T", reward == 1.0, f"reward {reward:.2f}"),
    ]

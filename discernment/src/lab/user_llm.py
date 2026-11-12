"""A realistic simulated user (E6): what the person does is drawn from a mix tied to the true outcome and a
persona; Gemini on Vertex AI then words it naturally for the exact purchase.

Behaviours (and what the labelling guide says they mean):
  praise 100 · accept ("ok, thanks") 80 · move_on (next task, no comment) 80 · adjust ("I wanted it in blue",
  a detail they had asked for) 20 · reject ("not what I asked", "not my style") 0 · repeat (asks again) 0 ·
  polite_unhappy ("thanks", then asks again) 0 · give_up 0 · settle ("I guess it'll do") unclear · silent unclear

The mix is fixed in code so every condition meets the same user, and drawn from the life's seeded rng so a
resumed day matches. Gemini only chooses the words; if a call fails, a plain fallback sentence is used.
The sense never saw these personas or Gemini's writing (its data is real chats plus Qwen-written gaps).
"""

import json
import os
import random
import time

from lab.user import Reaction

MIX = {  # behaviour weights per outcome band, before the persona shifts them
    "none": {"give_up": .5, "repeat": .3, "silent": .2},
    "wrong": {"reject": .5, "repeat": .2, "give_up": .1, "silent": .2},
    "partial": {"adjust": .4, "settle": .15, "reject": .1, "accept": .1, "polite_unhappy": .05, "silent": .2},
    "decent": {"accept": .3, "move_on": .2, "adjust": .2, "settle": .05, "polite_unhappy": .05, "silent": .2},
    "exact": {"praise": .4, "accept": .2, "move_on": .2, "silent": .2},
}
PERSONAS = {  # wording style, and how the persona shifts the mix
    "terse": ({}, "very short, few words, no pleasantries"),
    "chatty": ({}, "friendly and talkative, a couple of sentences"),
    "picky": ({"adjust": 1.6, "accept": 0.6, "move_on": 0.6, "praise": 0.7}, "precise and demanding"),
    "polite": ({"polite_unhappy": 2.0, "reject": 0.6}, "very polite, softens complaints"),
    "easygoing": ({"accept": 1.5, "move_on": 1.4, "adjust": 0.6, "settle": 1.3}, "relaxed and easily satisfied"),
    "blunt": ({"reject": 1.4, "give_up": 1.3, "polite_unhappy": 0.3}, "blunt, casual, may use slang"),
}
NEXT_MOVE = {"praise": "replied", "accept": "replied", "move_on": "new request", "adjust": "asked to adjust",
             "reject": "replied", "repeat": "repeated the request", "polite_unhappy": "repeated the request",
             "give_up": "left the session", "settle": "replied", "silent": "said nothing"}
FALLBACK = {"praise": "Perfect, thank you!", "accept": "Ok, thanks.", "adjust": "I wanted it with {miss}.",
            "reject": "That's not what I asked for.", "repeat": "{request}", "polite_unhappy": "Thanks! But {request}",
            "give_up": "Forget it.", "settle": "I guess it'll do."}
HOW = {
    "praise": "You are clearly happy: praise it or thank them warmly.",
    "accept": "You accept it without fuss, e.g. a short 'ok, thanks'. Do not praise it and do not complain.",
    "adjust": "Point out the one thing that missed what you asked for ({miss}). Say it as something you had asked for "
              "(e.g. 'I wanted it in...', 'I said...'), not as a new wish.",
    "reject": "Reject it: it is not what you asked for, or not your style. Do not ask for a small tweak.",
    "repeat": "Ask for the same thing again in your own words, as if the purchase did not happen: {request}",
    "polite_unhappy": "Say thanks, but then ask for the same thing again in other words: {request}",
    "give_up": "Give up on it, in a few words, as you leave.",
    "settle": "Reluctantly settle for it in a flat, lukewarm way that does not say whether it is good.",
}
MODEL = os.environ.get("USER_MODEL", "gemini-3-flash-preview")  # the Flash model this project can call (Nov 2026)


def band(outcome: dict) -> str:
    if not outcome["done"]:
        return "none"
    r = outcome["reward"]
    if r >= 1.0:
        return "exact"
    if outcome.get("info", {}).get("r_type", 1.0) < 0.5 or r < 0.5:
        return "wrong"
    return "decent" if r >= 0.75 else "partial"


def misses(outcome: dict) -> list[str]:
    """What the purchase got wrong against the hidden goal, in words (same checks as the template user)."""
    goal, bought = outcome["goal"], outcome["purchased"]
    opts = {o.lower() for o in bought.get("options") or []}
    out = [f"the {o} option" for o in goal["options"] if str(o).lower() not in opts]
    out += [f"something {a}" for a in goal["attributes"] if a.lower() not in (bought.get("text") or "")]
    if goal.get("price_upper") and bought.get("price") and bought["price"] > goal["price_upper"]:
        out.append(f"a price under ${goal['price_upper']}")
    return out


class LLMUser:
    def __init__(self, project: str | None = None):
        import requests

        self.http = requests.Session()
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or self._metadata("project/project-id")
        self.token, self.expires = None, 0.0

    def _metadata(self, path: str) -> str:
        return self.http.get(f"http://metadata.google.internal/computeMetadata/v1/{path}",
                             headers={"Metadata-Flavor": "Google"}, timeout=10).text

    def _auth(self) -> str:
        if time.time() > self.expires - 60:
            t = json.loads(self._metadata("instance/service-accounts/default/token"))
            self.token, self.expires = t["access_token"], time.time() + t["expires_in"]
        return self.token

    def react(self, outcome: dict, instruction: str, rng: random.Random) -> Reaction:
        persona = rng.choice(sorted(PERSONAS))
        shift, style = PERSONAS[persona]
        weights = {k: w * shift.get(k, 1.0) for k, w in MIX[band(outcome)].items()}
        kind = rng.choices(list(weights), weights=list(weights.values()))[0]
        if kind in ("silent", "move_on"):
            return Reaction(kind, "", NEXT_MOVE[kind], persona)
        miss = (rng.choice(misses(outcome)) if misses(outcome) else "what I described")
        what = HOW[kind].format(miss=miss, request=instruction)
        bought = outcome["purchased"]
        product = f"{(bought.get('title') or 'nothing')[:150]}, options {bought.get('options') or []}, ${bought.get('price')}"
        prompt = (f"You are a shopper who asked an AI assistant to buy something for you.\nYour request: {instruction}\n"
                  f"What it bought: {product if outcome['done'] else 'nothing'}\nYour personality: {style}.\n"
                  f"What you do now: {what}\nWrite only your next message to the assistant, in your own words, one or two "
                  'short sentences. Reply as JSON: {"said": "..."}')
        for attempt in range(2):  # one retry: Gemini occasionally returns cut-off JSON
            said = self._ask(prompt, rng)
            if said:
                return Reaction(kind, said, NEXT_MOVE[kind], persona)
        print("user model failed twice; plain fallback", flush=True)
        return Reaction(kind, FALLBACK[kind].format(miss=miss, request=instruction), NEXT_MOVE[kind], "fallback")

    def _ask(self, prompt: str, rng: random.Random) -> str:
        """Gemini's wording of the reaction, or "" if the call or its JSON failed."""
        try:
            r = self.http.post(
                f"https://aiplatform.googleapis.com/v1/projects/{self.project}/locations/global/publishers/google/models/{MODEL}:generateContent",
                headers={"Authorization": f"Bearer {self._auth()}"}, timeout=60, json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    # Gemini 3 thinks before answering; keep it light and leave room for the answer after it
                    "generationConfig": {"temperature": 0.9, "seed": rng.randrange(2**31), "maxOutputTokens": 600,
                                         "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}})
            r.raise_for_status()
            text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"] if not p.get("thought"))
            return json.loads(text)["said"].strip()
        except Exception as e:
            print(f"user model call failed ({e!r:.120})", flush=True)
            return ""

"""The feedback sense at work: scores a reaction through vLLM, from the first answer token's log-probabilities.

The sense is served as a LoRA on the same vLLM as the creature. Which adapter it reads through is the E5
manipulation: the sense's own (separate organ, f(W + dS)) or the creature's combined brain (coupled, f(W + dS + dL)).
A v2 sense carries calibration.json beside its adapter (a temperature and a threshold, from lab/sense/train2.py);
without one, v1's fixed 70% threshold applies. A v3 sense (version 3, lab/sense/train3.py) asks two questions:
is the reaction a clear verdict (if not: null), and then which score.
"""

import json
import math
from pathlib import Path
from functools import lru_cache

from lab.sense import CLEAR, LABELS, SCORES, prompt, prompt3

CONFIDENCE = 0.7  # v1: below this the answer counts as unclear


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


class SenseReader:
    def __init__(self, base_url: str, adapter_name: str = "sense", adapter_dir: str | Path | None = None):
        import requests

        self.http, self.base_url, self.adapter = requests.Session(), base_url.rstrip("/"), adapter_name
        cal = Path(adapter_dir) / "calibration.json" if adapter_dir else None
        cal = json.loads(cal.read_text()) if cal and cal.exists() else {}
        self.temperature, self.threshold = cal.get("temperature", 1.0), cal.get("threshold", CONFIDENCE)
        self.cal = cal
        tok = _tokenizer()
        # vLLM reports top tokens as decoded text, so match on each label's first token decoded the same way
        first = lambda labels: [tok.decode([tok(lab, add_special_tokens=False).input_ids[0]]) for lab in labels]
        self.label_tokens, self.clear_tokens, self.score_tokens = first(LABELS), first(CLEAR), first(SCORES)

    def _ask(self, messages: list[dict], tokens: list[str], temperature: float) -> list[float] | None:
        """One question's answer probabilities from the first answer token, or None if no answer is among the top."""
        text = _tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        r = self.http.post(f"{self.base_url}/v1/completions", timeout=120, json={
            "model": self.adapter, "prompt": text, "max_tokens": 1, "temperature": 0, "logprobs": 20})
        r.raise_for_status()
        top = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        logits = [top.get(t, -math.inf) for t in tokens]
        m = max(logits)
        if m == -math.inf:
            return None
        e = [math.exp((x - m) / temperature) for x in logits]
        return [x / sum(e) for x in e]

    def probs(self, row: dict) -> list[float]:
        """The five label probabilities (in LABELS order) for one row: request, next_move, said."""
        return self._ask(prompt(row), self.label_tokens, self.temperature) or [0, 0, 0, 0, 1.0]  # none: unclear

    def score(self, row: dict) -> tuple[str, float]:
        """The label and its confidence; anything under the threshold is null."""
        if self.cal.get("version") == 3:
            clear = self._ask(prompt3(row, "clear"), self.clear_tokens, self.cal["temperature_clear"])
            if not clear or clear[0] < self.cal["threshold_clear"]:
                return "null", clear[0] if clear else 0.0
            s = self._ask(prompt3(row, "score"), self.score_tokens, self.cal["temperature_score"])
            best = max(range(4), key=s.__getitem__) if s else 0
            return (SCORES[best] if s and s[best] >= self.cal["threshold_score"] else "null"), s[best] if s else 0.0
        p = self.probs(row)
        best = max(range(len(p)), key=p.__getitem__)
        return (LABELS[best] if p[best] >= self.threshold else "null"), p[best]

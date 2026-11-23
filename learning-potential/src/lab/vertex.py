"""Gemini on Vertex AI, called from a GCP VM with its own service account (no keys on disk).

Used as E7's judge (MemoryBench scores). lab/user_llm.py has its own copy of the same calls for E6's user.
"""

import json
import os
import time

MODEL = os.environ.get("JUDGE_MODEL", "gemini-3-flash-preview")


class Gemini:
    def __init__(self, model: str = MODEL, project: str | None = None):
        import requests

        self.http, self.model = requests.Session(), model
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

    def generate(self, system: str, user: str, json_mode: bool = True, temperature: float = 0.0, max_tokens: int = 1024,
                 thinking: str = "low", timeout: int = 120) -> str:
        """The model's text answer (thought parts left out). Raises on HTTP errors."""
        r = self.http.post(
            f"https://aiplatform.googleapis.com/v1/projects/{self.project}/locations/global/publishers/google/models/{self.model}:generateContent",
            headers={"Authorization": f"Bearer {self._auth()}"}, timeout=timeout, json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                # Gemini 3 thinks before answering; keep it light and leave room for the answer
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens,
                                     **({"responseMimeType": "application/json"} if json_mode else {}),
                                     "thinkingConfig": {"thinkingLevel": thinking}}})
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def content(self, body: dict, timeout: int = 300) -> dict:
        """One raw generateContent call (body: contents, tools, systemInstruction, generationConfig); returns the first
        candidate's content as is (parts with any functionCall and thoughtSignature, to send back unchanged)."""
        r = self.http.post(
            f"https://aiplatform.googleapis.com/v1/projects/{self.project}/locations/global/publishers/google/models/{self.model}:generateContent",
            headers={"Authorization": f"Bearer {self._auth()}"}, timeout=timeout, json=body)
        r.raise_for_status()
        return r.json()["candidates"][0].get("content") or {"role": "model", "parts": []}

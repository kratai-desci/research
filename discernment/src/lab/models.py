"""Model adapters. Every system's text is counted with the same tokenizer, so token numbers are comparable."""

import time
from dataclasses import dataclass
from functools import lru_cache

import ollama
from transformers import AutoTokenizer

TOKENIZER = "Qwen/Qwen3-0.6B"  # one tokenizer for counting every system's text


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def count_tokens(text: str) -> int:
    return len(_tokenizer()(text).input_ids) if text else 0


def count_messages(messages: list[dict]) -> int:
    text = _tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return count_tokens(text)


@dataclass
class Reply:
    content: str
    input_tokens: int
    latency_ms: int


class MLXModel:
    """A model run in-process with MLX, optionally with a LoRA adapter (the creature's learned part).
    MLX has no forced-JSON mode, so the agent parses the JSON leniently."""

    def __init__(self, name: str = "Qwen/Qwen3-0.6B", adapter: str | None = None, temperature: float = 0.0):
        self.name, self.adapter, self.temperature = name, adapter, temperature
        self._load()

    def _load(self):
        from mlx_lm import load

        self.model, self.tok = load(self.name, adapter_path=self.adapter)

    def set_adapter(self, adapter: str | None) -> None:
        """Swap in tonight's adapter (reloads the model)."""
        self.adapter = adapter
        self._load()

    def chat(self, messages: list[dict], schema: dict) -> Reply:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        start = time.perf_counter()
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text, depth, opened = "", 0, False
        for chunk in stream_generate(self.model, self.tok, prompt=prompt, max_tokens=300, sampler=make_sampler(temp=self.temperature)):
            text += chunk.text
            for ch in chunk.text:  # stop as soon as the first JSON object closes
                depth += ch == "{"
                opened |= ch == "{"
                depth -= ch == "}"
            if opened and depth <= 0:
                break
        return Reply(content=text, input_tokens=count_messages(messages), latency_ms=int((time.perf_counter() - start) * 1000))


class VLLMModel:
    """The same creature served by vLLM on a GPU (OpenAI-compatible server started with --enable-lora).
    Like MLXModel, JSON is not forced, so behaviour matches the Mac runs. Each night's adapter is loaded
    into the running server under its own name, so the server never restarts."""

    def __init__(self, name: str = "Qwen/Qwen3-0.6B", base_url: str = "http://127.0.0.1:8001", temperature: float = 0.0):
        import requests

        self.http, self.name, self.base_url, self.temperature = requests.Session(), name, base_url.rstrip("/"), temperature
        self.served = name  # the model name requests go to: the base, or tonight's adapter

    def set_adapter(self, adapter: str | None) -> None:
        if not adapter:
            self.served = self.name
            return
        lora_name = "creature-" + "-".join(adapter.rstrip("/").split("/")[-3:-1])  # e.g. creature-<life>-night_03
        r = self.http.post(f"{self.base_url}/v1/load_lora_adapter", json={"lora_name": lora_name, "lora_path": adapter}, timeout=300)
        if r.status_code != 200 and "already" not in r.text:
            raise RuntimeError(f"vLLM could not load adapter {adapter}: {r.status_code} {r.text}")
        if self.served != self.name and self.served != lora_name:  # drop yesterday's adapter: long lives load hundreds
            self.http.post(f"{self.base_url}/v1/unload_lora_adapter", json={"lora_name": self.served}, timeout=60)
        self.served = lora_name

    def chat(self, messages: list[dict], schema: dict, max_tokens: int = 300) -> Reply:
        start = time.perf_counter()
        r = self.http.post(f"{self.base_url}/v1/chat/completions", timeout=300, json={
            "model": self.served, "messages": messages, "temperature": self.temperature, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        if r.status_code == 400:  # e.g. a prompt longer than the server allows: count it as a failed step, don't end the life
            return Reply(content="", input_tokens=count_messages(messages), latency_ms=int((time.perf_counter() - start) * 1000))
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        return Reply(content=text, input_tokens=count_messages(messages), latency_ms=int((time.perf_counter() - start) * 1000))


class OllamaModel:
    """A frozen model served by Ollama. Thinking mode is off; reasoning goes in the JSON `thought` field."""

    def __init__(self, name: str):
        self.name = name

    def chat(self, messages: list[dict], schema: dict) -> Reply:
        start = time.perf_counter()
        response = ollama.chat(
            model=self.name,
            messages=messages,
            format=schema,
            think=False,
            options={"temperature": 0, "seed": 0, "num_ctx": 8192},
        )
        return Reply(
            content=response.message.content,
            input_tokens=count_messages(messages),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

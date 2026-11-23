"""E14: BFCL handlers for the two newest tiny models, which bfcl-eval (2026.3.23, the latest release) does not list,
and a launcher that registers them as models and runs the bfcl CLI.

    python -m lab.e14.new_handlers generate --model e14-qwen3.5-0.8b --test-category multi_turn ... (bfcl's flags)

Both handlers write the prompt with the model's own chat template (tokenizer.apply_chat_template, tools passed as
bfcl gives them), so the model sees exactly the format its makers trained it on, and read the model's own tool-call
format:
    Qwen3.5      <tool_call><function=name><parameter=p>value</parameter>...</function></tool_call>, values typed
                 from the function's schema (a string parameter stays a string)
    LFM2.5       <|tool_call_start|>[name(p=value, ...)]<|tool_call_end|>, Python literals (the markers may be
                 dropped as special tokens: a reply that is only such a list is read the same way)
The calls found are what bfcl stores as the model's response (a JSON list of {name, arguments}), which decode_ast
and decode_execute read back. Needs a vLLM and transformers recent enough for these architectures (qwen3_5, lfm2):
run_bfcl_arms.sh with NEW_STACK=1 builds that venv.
"""
import ast
import json
import os
import re
import sys
import threading
from typing import Any

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override

_local = threading.local()  # the current question's functions, for typing Qwen3.5's parameter values


def _types(functions: list) -> dict:
    out = {}
    for f in functions or []:
        f = f.get("function", f)
        props = ((f.get("parameters") or {}).get("properties") or {})
        out[f.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
    return out


class ChatTemplateHandler(OSSHandler):
    """Prompt by the model's own chat template; subclasses read the calls out of a reply."""

    def __init__(self, model_name, temperature, registry_name, is_fc_model, dtype="bfloat16", **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name

    def calls(self, text: str) -> list[dict]:
        raise NotImplementedError

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        return {"message": [], "function": test_entry["function"]}

    @override
    def _format_prompt(self, messages, function):
        _local.types = _types(function)
        chat = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                chat.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                    {"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                    for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                chat.append({"role": "tool", "content": str(m.get("content", ""))})
            else:
                chat.append({"role": m["role"], "content": m.get("content", "")})
        tools = [t if "function" in t else {"type": "function", "function": t} for t in (function or [])]
        return self.tokenizer.apply_chat_template(chat, tools=tools or None, add_generation_prompt=True, tokenize=False)

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        text = api_response.choices[0].text
        reasoning, answer = "", text
        if "</think>" in text:
            reasoning, answer = text.split("</think>", 1)
            reasoning = reasoning.split("<think>")[-1].strip()
            answer = answer.lstrip("\n")
        found = self.calls(answer)
        if found:
            stored = json.dumps(found)
            history = {"role": "assistant", "content": "", "tool_calls": found}
        else:
            stored = answer
            history = {"role": "assistant", "content": answer}
        return {"model_responses": stored, "reasoning_content": reasoning,
                "model_responses_message_for_chat_history": history,
                "input_token": api_response.usage.prompt_tokens, "output_token": api_response.usage.completion_tokens}

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data

    @staticmethod
    def _stored(result) -> list[dict]:
        try:
            calls = json.loads(result) if isinstance(result, str) else result
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(calls, list) or not all(isinstance(c, dict) and "name" in c for c in calls):
            return []
        return calls

    @override
    def decode_ast(self, result, language="Python", has_tool_call_tag=False):
        return [{c["name"]: dict(c.get("arguments") or {})} for c in self._stored(result)]

    @override
    def decode_execute(self, result, has_tool_call_tag=False):
        return convert_to_function_call([{c["name"]: dict(c.get("arguments") or {})} for c in self._stored(result)])


class Qwen35Handler(ChatTemplateHandler):
    CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
    PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)

    @override
    def calls(self, text: str) -> list[dict]:
        types = getattr(_local, "types", {})
        out = []
        for name, body in self.CALL.findall(text):
            args = {}
            for p, raw in self.PARAM.findall(body):
                kind = (types.get(name) or {}).get(p)
                if kind == "string":
                    args[p] = raw
                    continue
                try:  # numbers, booleans, lists, objects are written as JSON
                    args[p] = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        args[p] = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        args[p] = raw
            out.append({"name": name, "arguments": args})
        return out


class LFM25Handler(ChatTemplateHandler):
    SPAN = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.S)

    @staticmethod
    def _pythonic(src: str) -> list[dict]:
        try:
            tree = ast.parse(src.strip(), mode="eval").body
        except SyntaxError:
            return []
        items = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
        out = []
        for c in items:
            if not isinstance(c, ast.Call):
                return []
            name = ast.unparse(c.func)  # dotted names stay dotted
            try:
                args = {k.arg: ast.literal_eval(k.value) for k in c.keywords if k.arg}
            except (ValueError, SyntaxError):
                return []
            out.append({"name": name, "arguments": args})
        return out

    @override
    def calls(self, text: str) -> list[dict]:
        spans = self.SPAN.findall(text)
        if spans:
            return [c for s in spans for c in self._pythonic(s)]
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):  # the markers dropped as special tokens
            return self._pythonic(stripped)
        return []


def register():
    """The two newest tiny models as bfcl models, with their own chat templates and tool-call formats."""
    for key, hf, display, handler, url, lic in [
        ("e14-qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B (own template; E14)", Qwen35Handler,
         "https://huggingface.co/Qwen/Qwen3.5-0.8B", "apache-2.0"),
        ("e14-lfm2.5-1.2b", "LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5-1.2B-Instruct (own template; E14)", LFM25Handler,
         "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct", "lfm1.0")]:
        MODEL_CONFIG_MAPPING[key] = ModelConfig(
            model_name=hf, display_name=display, url=url, org=hf.split("/")[0], license=lic, model_handler=handler,
            input_price=None, output_price=None, is_fc_model=True, underscore_to_dot=False)


if __name__ == "__main__":
    register()
    from bfcl_eval.__main__ import cli
    sys.argv[0] = "bfcl"
    cli()

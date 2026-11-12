#!/bin/bash
# τ²-bench as Experiment 5 ran it: the repository pinned at commit 033bf08, with Gemini 3 Flash (on Vertex AI) in
# place of the published GPT models for the graders as well as the customer. The customer model is passed by
# lab.life_tau2 itself (--user-llm); this script switches the three grader settings in τ²-bench's config.py.
#   TAU2_DIR=~/tau2-bench bash src/setup/tau2_gemini.sh
# Needs uv and Vertex AI credentials (VERTEXAI_PROJECT, VERTEXAI_LOCATION=global) for the account that runs the lives.
set -euo pipefail
TAU2_DIR="${TAU2_DIR:-$HOME/tau2-bench}"
MODEL="vertex_ai/gemini-3-flash-preview"

if [ ! -x "$TAU2_DIR/.venv/bin/tau2" ]; then
  git clone -q https://github.com/sierra-research/tau2-bench.git "$TAU2_DIR"
  git -C "$TAU2_DIR" checkout -q 033bf08
  (cd "$TAU2_DIR" && uv sync -q && uv pip install -q google-cloud-aiplatform)
fi

# The natural-language assertion grader (temperature 0), the environment interface and the semantic
# communication check all on Gemini. (sed -i '' on macOS.)
sed -i \
  -e "s|^DEFAULT_LLM_NL_ASSERTIONS = .*|DEFAULT_LLM_NL_ASSERTIONS = \"$MODEL\"|" \
  -e 's|^DEFAULT_LLM_NL_ASSERTIONS_ARGS = .*|DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0}|' \
  -e "s|^DEFAULT_LLM_ENV_INTERFACE = .*|DEFAULT_LLM_ENV_INTERFACE = \"$MODEL\"|" \
  -e "s|^DEFAULT_SEMANTIC_COMMUNICATE_MODEL = .*|DEFAULT_SEMANTIC_COMMUNICATE_MODEL = \"$MODEL\"|" \
  "$TAU2_DIR/src/tau2/config.py"
grep -E "^DEFAULT_(LLM_NL_ASSERTIONS|LLM_ENV_INTERFACE|SEMANTIC_COMMUNICATE_MODEL)" "$TAU2_DIR/src/tau2/config.py"

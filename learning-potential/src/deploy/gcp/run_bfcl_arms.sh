#!/bin/bash
# On an L4 VM, from the repo folder: BFCL v4 exams for several models ("arms") one after another, for E12.
#   RUN=e12 ARMS="qwen25-1.5b|Qwen/Qwen3-1.7B-FC|hf:Qwen/Qwen2.5-1.5B-Instruct xlam2-1b|Salesforce/xLAM-2-1b-fc-r|" \
#     nohup bash deploy/gcp/run_bfcl_arms.sh > logs/bfcl-arms.log 2>&1 &
# Each arm is name|bfcl model (the handler and prompt format)|weights, where weights is empty (bfcl's own model),
# hf:<repo> (downloaded) or gs://... (a model folder in the bucket, e.g. our fine-tuned Qwen3). Models bfcl does not
# list (Qwen2.5, ours) run through a listed model with the same chat format: Qwen2.5 and Qwen3 share Qwen's
# <tool_call> format, so both use Qwen/Qwen3-1.7B-FC's handler.
# Per arm: ~/bfcl-$RUN/<name>/ holds bfcl's results and scores; $BUCKET/bfcl/$RUN/<name>.json its summary
# (lab/bfcl_summary.py) and $BUCKET/bfcl/$RUN/<name>/ the raw files. web_search is left out (paid search key).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
RUN="${RUN:-e12}"; ARMS="${ARMS:?set ARMS}"
# E13: PRACTICE=gs://.../e13/practice runs the practice set (lab/e13/practice.py) as multi_turn_base instead of bfcl's own
PRACTICE="${PRACTICE:-}"
if [ -n "$PRACTICE" ]; then CATEGORIES="${CATEGORIES:-multi_turn_base}"; else CATEGORIES="${CATEGORIES:-single_turn multi_turn memory}"; fi
PROJECT=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
BUCKET="${BUCKET:-gs://$PROJECT-alife}"
# NEW_STACK=1: a second venv with the latest vLLM and transformers, for architectures vLLM 0.8.5 cannot load (E14's
# Qwen3.5 and LFM2.5); bfcl-eval itself is the same version
if [ -n "${NEW_STACK:-}" ]; then VENV="${VENV:-$HOME/bfcl-new-venv}"; else VENV="${VENV:-$HOME/bfcl-venv}"; fi
REPO="$PWD"
mkdir -p logs "$HOME/models"

# bfcl-eval as in E11 (deploy/gcp/run_bfcl.sh)
[ -f /usr/include/python3.10/Python.h ] || sudo apt-get install -yq python3.10-dev
if [ ! -x "$VENV/bin/bfcl" ]; then
  if [ -n "${NEW_STACK:-}" ]; then
    # bfcl here has no vLLM (its dependencies would pull vLLM back to 0.8.5); vLLM runs from its own venv below
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" "bfcl-eval==2026.3.23" transformers soundfile setuptools
  else
    uv venv --python 3.10 "$VENV"
    uv pip install --python "$VENV/bin/python" "bfcl-eval[oss_eval_vllm]==2026.3.23" "transformers==4.51.3" soundfile setuptools
  fi
fi
VLLM_VENV="$HOME/vllm-new-venv"
if [ -n "${NEW_STACK:-}" ] && [ ! -x "$VLLM_VENV/bin/vllm" ]; then
  uv venv --python 3.12 "$VLLM_VENV"
  uv pip install --python "$VLLM_VENV/bin/python" "vllm>=0.11" ninja  # vLLM compiles kernels at start-up
fi
export BFCL_VERSION=$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('bfcl-eval'))")

status() {  # root arm stage message
  "$VENV/bin/python" "$REPO/lab/bfcl_summary.py" "$1" "$3" "$4" > "$1/e11.tmp" && mv "$1/e11.tmp" "$1/e11.json" \
    && gcloud storage cp -q "$1/e11.json" "$BUCKET/bfcl/$RUN/$2.json"
}

if [ -n "$PRACTICE" ]; then
  waited=0
  until gcloud storage ls "$PRACTICE/possible_answer/BFCL_v4_multi_turn_base.json" > /dev/null 2>&1; do
    [ $waited -ge $(( ${WAIT_HOURS:-48} * 12 )) ] && { echo "no practice set at $PRACTICE"; exit 1; }
    sleep 300; waited=$((waited + 1))
  done
  export E13_DATA="$HOME/e13-$(basename "$PRACTICE")"; mkdir -p "$E13_DATA"  # one folder per set version
  gcloud storage cp -r -q "$PRACTICE/*" "$E13_DATA/"
fi

for arm in $ARMS; do
  IFS='|' read -r name model weights <<< "$arm"
  # toolrl-* models are ToolRL's own exam (lab/e12/toolrl_bfcl.py registers them, then runs the bfcl CLI)
  case "$model" in toolrl-*) BFCL="python -m lab.e12.toolrl_bfcl" ;; e14-*) BFCL="python -m lab.e14.new_handlers" ;; *) BFCL="bfcl" ;; esac
  [ -n "$PRACTICE" ] && BFCL="python -m lab.e13.practice_bfcl"
  # E18: the exam with a learned state in the prompt (lab/e18/exam_bfcl.py); it registers the E14 handlers itself
  [ -n "${BFCL_CMD:-}" ] && BFCL="$BFCL_CMD"
  root="$HOME/bfcl-$RUN/$name"; mkdir -p "$root"
  export BFCL_PROJECT_ROOT="$root"
  local_path=()
  case "$weights" in
    local:*) dir="${weights#local:}"; local_path=(--local-model-path "$dir") ;;  # E18: a merged model already here
    hf:*) dir="$HOME/models/$name"
          [ -f "$dir/config.json" ] || "$VENV/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "${weights#hf:}" "$dir" > /dev/null
          local_path=(--local-model-path "$dir") ;;
    gs://*) dir="$HOME/models/$name"
          # a model still being trained: wait for it to be uploaded, up to WAIT_HOURS (each object appears only once
          # complete, so the config and the weights both being there means the upload is through)
          ready() { gcloud storage ls "$weights/config.json" > /dev/null 2>&1 && gcloud storage ls "$weights/model*.safetensors" > /dev/null 2>&1 \
                    && gcloud storage ls "$weights/tokenizer_config.json" > /dev/null 2>&1; }
          waited=0
          until ready; do
            [ $waited -ge $(( ${WAIT_HOURS:-48} * 12 )) ] && break
            [ $waited -eq 0 ] && status "$root" "$name" waiting "$name: waiting for its model ($weights)"
            sleep 300; waited=$((waited + 1))
          done
          ready || { status "$root" "$name" failed "$name: no model at $weights after ${WAIT_HOURS:-48} h"; continue; }
          [ -f "$dir/config.json" ] || { mkdir -p "$dir"; gcloud storage cp -r -q "$weights/*" "$dir/"; }
          local_path=(--local-model-path "$dir") ;;
  esac
  # MAX_LEN caps a local model's context (vLLM sizes its cache for the declared length, e.g. 262k for
  # Qwen3-4B-Instruct-2507, which does not fit a 24 GB L4)
  if [ -n "${MAX_LEN:-}" ] && [ ${#local_path[@]} -gt 0 ]; then
    "$VENV/bin/python" - "${local_path[1]}/config.json" "$MAX_LEN" <<'PY'
import json, sys
p, n = sys.argv[1], int(sys.argv[2]); c = json.load(open(p))
changed = False
for part in (c, c.get("text_config") or {}):  # multimodal configs keep it under text_config
    if part.get("max_position_embeddings", 0) > n:
        part["max_position_embeddings"] = n; changed = True
if changed:
    json.dump(c, open(p, "w"), indent=2)
# bfcl falls back on the tokenizer's model_max_length when the config has no top-level max_position_embeddings
t = p.replace("config.json", "tokenizer_config.json")
try:
    tc = json.load(open(t))
    if tc.get("model_max_length", 0) > n:
        tc["model_max_length"] = n; json.dump(tc, open(t, "w"), indent=2)
except (OSError, ValueError):
    pass
PY
  fi
  server=""; skip=()
  if [ -n "${NEW_STACK:-}" ]; then
    # the model served by the new vLLM under its local path (the name bfcl asks for), on bfcl's default port
    "$VLLM_VENV/bin/python" -c "import ninja" 2>/dev/null || uv pip install --python "$VLLM_VENV/bin/python" ninja
    PATH="$VLLM_VENV/bin:/usr/local/cuda/bin:$PATH" "$VLLM_VENV/bin/vllm" serve "${local_path[1]}" --port 1053 --dtype bfloat16 --gpu-memory-utilization 0.9 \
      --max-model-len "${MAX_LEN:-40960}" --trust-remote-code > "$REPO/logs/vllm-$name.log" 2>&1 &
    server=$!
    for k in $(seq 1 120); do curl -sf localhost:1053/health > /dev/null && break; sleep 10; done
    curl -sf localhost:1053/health > /dev/null || { status "$root" "$name" failed "$name: vLLM server did not start (logs/vllm-$name.log)"; kill $server 2>/dev/null; continue; }
    skip=(--skip-server-setup)
  fi
  for cat in $CATEGORIES; do
    status "$root" "$name" generating "$name ($model): $cat"
    (cd "$root" && PATH="$VENV/bin:/usr/local/cuda/bin:$PATH" PYTHONPATH="$REPO" $BFCL generate --model "$model" --test-category "$cat" \
      --backend vllm --num-gpus 1 --gpu-memory-utilization 0.9 "${local_path[@]}" "${skip[@]}" >> "$REPO/logs/bfcl-$name.log" 2>&1) \
      || echo "generate failed: $name $cat"
  done
  [ -n "$server" ] && { kill $server 2>/dev/null; wait $server 2>/dev/null; }
  status "$root" "$name" evaluating "$name"
  (cd "$root" && PATH="$VENV/bin:$PATH" PYTHONPATH="$REPO" $BFCL evaluate --model "$model" --test-category "${CATEGORIES// /,}" >> "$REPO/logs/bfcl-$name.log" 2>&1) \
    || echo "evaluate failed: $name"
  if ls "$root"/score/*/*/*_score.json > /dev/null 2>&1; then
    status "$root" "$name" done "finished ($model, weights ${weights:-bfcl default})"
    gcloud storage cp -r -q "$root/score" "$root/result" "$BUCKET/bfcl/$RUN/$name/" 2>/dev/null
  else
    status "$root" "$name" failed "no scores: see logs/bfcl-$name.log on the VM"
  fi
done
[ -n "${NO_SHUTDOWN:-}" ] || sudo shutdown -h now

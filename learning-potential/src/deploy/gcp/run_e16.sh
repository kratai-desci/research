#!/bin/bash
# E16 on an L4 VM (spot or regular), from the repo folder: LoRA fine-tuning on the job (lab/e16/lora.py). One run
# per VM by default (training needs room on the GPU beside vLLM); several runs of a fast model can share a VM:
#   MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUNS="teach-s15|teach|15" TASKS=1200 \
#     nohup bash deploy/gcp/run_e16.sh > logs/e16.log 2>&1 &
# MODEL and RUNS as in run_e15.sh. vLLM serves the base model with LoRA enabled and takes about half the GPU; each
# run trains its adapter in the rest and hot-loads it into vLLM after every update.
# Results: $BUCKET/e16/<name>/<run>/ (status.json every minute; log, adapters and state every 5 minutes).
#
# Safe against errors, and resumable: each run saves its state at every update (every 10 tasks) and resumes from it;
# a run that stops is restarted, up to 5 times; the run folders are backed up to the bucket and restored on start,
# so a reclaimed spot VM, or any VM, continues where it stopped.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
EXP="${EXP:-e16}"  # E19 runs this same code under its own name
MODEL="${MODEL:?set MODEL}"; RUNS="${RUNS:?set RUNS}"; TASKS="${TASKS:-1200}"
IFS='|' read -r name registry weights <<< "$MODEL"
PROJECT=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
BUCKET="${BUCKET:-gs://$PROJECT-alife}"
REPO="$PWD"; DATA="$HOME/e15-data"; OUT="$HOME/$EXP/$name"
mkdir -p logs "$DATA" "$OUT" "$HOME/models"
if [ -z "${HF_TOKEN:-}" ]; then  # gated models: the token comes from Secret Manager, never printed
  HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token --project "$PROJECT" 2>/dev/null) && export HF_TOKEN
fi

# venvs as in E14 (run_bfcl_arms.sh)
[ -f /usr/include/python3.10/Python.h ] || sudo apt-get install -yq python3.10-dev
if [ -n "${NEW_STACK:-}" ]; then
  VENV="$HOME/bfcl-new-venv"; VLLM="$HOME/vllm-new-venv/bin/vllm"
  [ -x "$VENV/bin/bfcl" ] || { uv venv --python 3.12 "$VENV"; uv pip install --python "$VENV/bin/python" "bfcl-eval==2026.3.23" transformers soundfile setuptools; }
  [ -x "$VLLM" ] || { uv venv --python 3.12 "$HOME/vllm-new-venv"; uv pip install --python "$HOME/vllm-new-venv/bin/python" "vllm>=0.11" ninja; }
else
  VENV="$HOME/bfcl-venv"; VLLM="$VENV/bin/vllm"
  [ -x "$VENV/bin/bfcl" ] || { uv venv --python 3.10 "$VENV"; uv pip install --python "$VENV/bin/python" "bfcl-eval[oss_eval_vllm]==2026.3.23" "transformers==4.51.3" soundfile setuptools; }
fi

# the E13 pool: v3, v4, v5
for v in practice-v3 practice-v4 practice-v5; do
  [ -s "$DATA/$v/possible_answer/BFCL_v4_multi_turn_base.json" ] || {
    mkdir -p "$DATA/$v/possible_answer"
    gcloud storage cp -q "$BUCKET/e13/$v/BFCL_v4_multi_turn_base.json" "$DATA/$v/"
    gcloud storage cp -q "$BUCKET/e13/$v/possible_answer/BFCL_v4_multi_turn_base.json" "$DATA/$v/possible_answer/"; }
done

# the weights: bfcl's own model (served under its Hugging Face name) or a downloaded one (served under its folder)
serve="$("$VENV/bin/python" -c "import sys
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING as M
k = sys.argv[1]
if k.startswith('e14-'):
    sys.path.insert(0, '$REPO'); from lab.e14.new_handlers import register; register()
print(M[k].model_name)" "$registry")"
path_arg=()
case "$weights" in
  hf:*) serve="$HOME/models/$name"
        [ -f "$serve/config.json" ] || "$VENV/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "${weights#hf:}" "$serve" > /dev/null
        path_arg=(--model-path "$serve") ;;
esac
[ -n "${MAX_LEN:-}" ] && [ ${#path_arg[@]} -gt 0 ] && "$VENV/bin/python" - "$serve" "$MAX_LEN" <<'PY'
import json, sys
d, n = sys.argv[1], int(sys.argv[2])
c = json.load(open(f"{d}/config.json"))
for part in (c, c.get("text_config") or {}):
    if part.get("max_position_embeddings", 0) > n:
        part["max_position_embeddings"] = n
json.dump(c, open(f"{d}/config.json", "w"), indent=2)
try:
    t = json.load(open(f"{d}/tokenizer_config.json"))
    if t.get("model_max_length", 0) > n:
        t["model_max_length"] = n; json.dump(t, open(f"{d}/tokenizer_config.json", "w"), indent=2)
except (OSError, ValueError):
    pass
PY

# the trainer's packages (the bfcl venv of the old stack has torch through vLLM; the new stack's has none)
if [ -n "${NEW_STACK:-}" ]; then
  "$VENV/bin/python" -c "import torch, peft" 2>/dev/null || uv pip install --python "$VENV/bin/python" torch peft
else
  "$VENV/bin/python" -c "import peft" 2>/dev/null || uv pip install --python "$VENV/bin/python" "peft==0.15.2"
fi

# restore earlier work, then back it up: status every minute, everything every 5 minutes
for r in $RUNS; do
  IFS='|' read -r run _ _ <<< "$r"
  [ -f "$OUT/$run/state.json" ] || gcloud storage cp -r -q "$BUCKET/$EXP/$name/$run" "$OUT/" 2>/dev/null
done
( n=0
  while sleep 60; do
    n=$((n + 1))
    for d in "$OUT"/*/; do
      run=$(basename "$d")
      [ -f "$d/status.json" ] && gcloud storage cp -q "$d/status.json" "$BUCKET/$EXP/$name/$run/status.json"
      [ $((n % 5)) -eq 0 ] && gcloud storage rsync -r -q -x '.*\.tmp$' "$d" "$BUCKET/$EXP/$name/$run"
    done
  done ) &
backup=$!

# vLLM with LoRA, adapters loaded and unloaded at run time, on about half the GPU
[ -n "${NEW_STACK:-}" ] && export VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True PATH="$(dirname "$VLLM"):/usr/local/cuda/bin:$PATH" "$VLLM" serve "$serve" \
  --port 1053 --dtype bfloat16 --gpu-memory-utilization "${GPU_UTIL:-0.45}" ${MAX_LEN:+--max-model-len "$MAX_LEN"} \
  --enable-lora --max-lora-rank 16 --max-loras 2 --trust-remote-code > "logs/vllm-e16-$name.log" 2>&1 &
server=$!
for k in $(seq 1 120); do curl -sf localhost:1053/health > /dev/null && break; sleep 10; done
curl -sf localhost:1053/health > /dev/null || { echo "vLLM did not start: logs/vllm-e16-$name.log"; kill $server $backup; exit 1; }

# the runs, side by side; each restarted up to 5 times (it resumes from its state)
pids=()
for r in $RUNS; do
  IFS='|' read -r run arm seed n <<< "$r"
  ( for try in 1 2 3 4 5; do
      grep -q '"stage": "done"' "$OUT/$run/status.json" 2>/dev/null && break
      PYTHONPATH="$REPO" "$VENV/bin/python" -m lab.e16.lora --registry "$registry" "${path_arg[@]}" \
        --data "$DATA/practice-v3" "$DATA/practice-v4" "$DATA/practice-v5" --out "$OUT/$run" --arm "$arm" \
        --seed "$seed" --tasks "${n:-$TASKS}" ${ARGS:-} >> "logs/e16-$name-$run.log" 2>&1 && break
      echo "$(date) $run stopped (try $try)" >> "logs/e16-$name-$run.log"; sleep 30
    done ) &
  pids+=($!)
done
wait "${pids[@]}"

kill $server $backup 2>/dev/null
for d in "$OUT"/*/; do gcloud storage rsync -r -q -x '.*\.tmp$' "$d" "$BUCKET/$EXP/$name/$(basename "$d")"; done
[ -n "${NO_SHUTDOWN:-}" ] || sudo shutdown -h now

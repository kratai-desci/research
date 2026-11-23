#!/bin/bash
# E17 probes on a VM, from the repo folder: every saved memory snapshot of a model's runs (memory-<n>.jsonl, every 100
# tasks) scored on the fixed probe tasks (lab/e15/probe.py --memory), as they appear, on its own VM:
#   MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUNS="teach-s15 judge-s15 teach600-s15" nohup bash deploy/gcp/run_e17_probe.sh > logs/e17-probe.log 2>&1 &
# Results: $BUCKET/e17/<name>/<run>/probe-<n>.json; checkpoint 0 is E15's (the same base model, no examples). A run
# whose name holds "600" is probed with the 600-token budget it ran with. A score already in the bucket is never redone.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
MODEL="${MODEL:?set MODEL}"; RUNS="${RUNS:?set RUNS}"; 
IFS='|' read -r name registry weights <<< "$MODEL"
PROJECT=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
BUCKET="${BUCKET:-gs://$PROJECT-alife}"
REPO="$PWD"; DATA="$HOME/e15-data"; OUT="$HOME/e17p/$name"
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

[ -n "${NEW_STACK:-}" ] && export VLLM_USE_FLASHINFER_SAMPLER=0
PATH="$(dirname "$VLLM"):/usr/local/cuda/bin:$PATH" "$VLLM" serve "$serve" --port 1053 --dtype bfloat16 \
  --gpu-memory-utilization 0.9 ${MAX_LEN:+--max-model-len "$MAX_LEN"} --trust-remote-code > "logs/vllm-e17p-$name.log" 2>&1 &
server=$!
for k in $(seq 1 120); do curl -sf localhost:1053/health > /dev/null && break; sleep 10; done
curl -sf localhost:1053/health > /dev/null || { echo "vLLM did not start: logs/vllm-e17p-$name.log"; kill $server; exit 1; }

probe() {  # memory snapshot, budget, and where its score goes in the bucket
  local memory=$1 budget=$2 dest=$3 tmp="$OUT/probe-tmp.json"
  gcloud storage ls "$dest" > /dev/null 2>&1 && return 0
  for try in 1 2 3; do
    PYTHONPATH="$REPO" "$VENV/bin/python" -m lab.e15.probe --registry "$registry" "${path_arg[@]}" \
      --data "$DATA/practice-v3" "$DATA/practice-v4" "$DATA/practice-v5" --memory "$memory" --budget "$budget" \
      --out "$tmp" >> "logs/e17p-$name.log" 2>&1 && gcloud storage cp -q "$tmp" "$dest" && return 0
    echo "$(date) probe failed ($memory, try $try)" >> "logs/e17p-$name.log"; sleep 30
  done
}

for round in $(seq 1 600); do  # at most ~50 hours of waiting for snapshots
  pending=0
  for run in $RUNS; do
    mkdir -p "$OUT/$run"
    for n in $(seq 100 100 1200); do
      dest="$BUCKET/e17/$name/$run/probe-$n.json"
      gcloud storage ls "$dest" > /dev/null 2>&1 && continue
      if gcloud storage cp -q "$BUCKET/e17/$name/$run/memory-$n.jsonl" "$OUT/$run/memory-$n.jsonl" 2>/dev/null; then
        case "$run" in *600*) b=600 ;; *) b=1200 ;; esac
        probe "$OUT/$run/memory-$n.jsonl" "$b" "$dest"
      else
        pending=1  # not saved yet: the run has not reached task n
      fi
    done
  done
  [ $pending -eq 0 ] && break
  sleep 300
done
kill $server 2>/dev/null
[ -n "${NO_SHUTDOWN:-}" ] || sudo shutdown -h now

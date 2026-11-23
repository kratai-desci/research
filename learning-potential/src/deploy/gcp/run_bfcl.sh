#!/bin/bash
# On an L4 VM, from the repo folder: E11, the BFCL v4 baseline. Plain Qwen3 models (no learning) through the
# official bfcl-eval package, to check our test setup reproduces the published leaderboard numbers.
#   nohup bash deploy/gcp/run_bfcl.sh > logs/bfcl.log 2>&1 &
# MODELS and CATEGORIES can be overridden. web_search is left out (it needs a paid SerpAPI key).
# Progress and scores go to $BUCKET/bfcl/e11.json (read by the dashboard); raw results and scores to $BUCKET/bfcl/e11/.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
MODELS="${MODELS:-Qwen/Qwen3-0.6B-FC Qwen/Qwen3-1.7B-FC}"
CATEGORIES="${CATEGORIES:-single_turn multi_turn memory}"
PROJECT=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
BUCKET="${BUCKET:-gs://$PROJECT-alife}"
export BFCL_PROJECT_ROOT="$HOME/bfcl-e11"
VENV="$HOME/bfcl-venv"
mkdir -p logs "$BFCL_PROJECT_ROOT"
REPO="$PWD"

status() {  # stage message: summarise whatever scores exist, then upload
  "$VENV/bin/python" "$REPO/lab/bfcl_summary.py" "$BFCL_PROJECT_ROOT" "$1" "$2" > "$BFCL_PROJECT_ROOT/e11.tmp" \
    && mv "$BFCL_PROJECT_ROOT/e11.tmp" "$BFCL_PROJECT_ROOT/e11.json" && gcloud storage cp -q "$BFCL_PROJECT_ROOT/e11.json" "$BUCKET/bfcl/e11.json"
}

# bfcl-eval with its vLLM extra, in its own venv (it pins vLLM 0.8.5, which needs transformers 4.51, not 5.x;
# its Qwen handler imports qwen-agent, which needs soundfile; vLLM's Triton compiles against Python.h and needs
# setuptools at start-up)
[ -f /usr/include/python3.10/Python.h ] || sudo apt-get install -yq python3.10-dev
if [ ! -x "$VENV/bin/bfcl" ]; then
  uv venv --python 3.10 "$VENV"
  uv pip install --python "$VENV/bin/python" "bfcl-eval[oss_eval_vllm]==2026.3.23" "transformers==4.51.3" soundfile setuptools
fi
VERSION=$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('bfcl-eval'))")
echo "bfcl-eval $VERSION"
export BFCL_VERSION="$VERSION"
status installing "bfcl-eval $VERSION installed"

cd "$BFCL_PROJECT_ROOT"
for model in $MODELS; do
  for cat in $CATEGORIES; do
    status generating "$model: $cat"
    PATH="$VENV/bin:/usr/local/cuda/bin:$PATH" bfcl generate --model "$model" --test-category "$cat" \
      --backend vllm --num-gpus 1 --gpu-memory-utilization 0.9 >> "$REPO/logs/bfcl-generate.log" 2>&1 \
      || echo "generate failed: $model $cat"
  done
  status evaluating "$model"
  PATH="$VENV/bin:$PATH" bfcl evaluate --model "$model" --test-category "${CATEGORIES// /,}" >> "$REPO/logs/bfcl-evaluate.log" 2>&1 \
    || echo "evaluate failed: $model"
done

# multi-turn a second time for the 1.7B model, to see whether its scores are stable from run to run
if [ -z "${NO_REPEAT:-}" ]; then
  status repeating "multi_turn again (stability check)"
  mv result result-run1 && mv score score-run1
  m=$(echo $MODELS | awk '{print $NF}')
  PATH="$VENV/bin:/usr/local/cuda/bin:$PATH" bfcl generate --model "$m" --test-category multi_turn \
    --backend vllm --num-gpus 1 --gpu-memory-utilization 0.9 >> "$REPO/logs/bfcl-generate.log" 2>&1
  PATH="$VENV/bin:$PATH" bfcl evaluate --model "$m" --test-category multi_turn >> "$REPO/logs/bfcl-evaluate.log" 2>&1
  mv result result-repeat && mv score score-repeat && mv result-run1 result && mv score-run1 score
fi

status done "finished"
gcloud storage cp -r -q result score "$BUCKET/bfcl/e11/" 2>/dev/null
[ -d score-repeat ] && gcloud storage cp -r -q score-repeat "$BUCKET/bfcl/e11/" 2>/dev/null
# stay up when nothing was scored, so the logs can be read
ls score/*/*/*_score.json >/dev/null 2>&1 || { status failed "no scores: see logs/bfcl-generate.log on the VM"; exit 1; }
[ -n "${NO_SHUTDOWN:-}" ] || sudo shutdown -h now

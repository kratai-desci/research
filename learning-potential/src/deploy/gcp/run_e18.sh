#!/bin/bash
# E18 on a VM, from the repo folder: the official BFCL v4 exam (all categories, as E14's floors) for the final state
# (task 1,200) of a model's E15-E17 runs, one exam after another:
#   MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" EXAMS="e15|teach-s15 e16|teach-s15 e17|teach600-s15" \
#     nohup bash deploy/gcp/run_e18.sh > logs/e18.log 2>&1 &
# MODEL as in run_e15.sh (NEW_STACK and MAX_LEN too). Each exam goes through E14's pipeline (run_bfcl_arms.sh):
#   e15: the final notes in every question's prompt (lab/e18/exam_bfcl.py), as in practice;
#   e16: the final adapter merged into the weights (lab/e18/merge.py), examined like a floor (e19's runs too);
#   e17: examples retrieved from the final memory in every question's prompt (lab/e18/exam_bfcl.py).
# Results: $BUCKET/bfcl/e18/<exp>-<model>-<run>.json, in E14's format. An exam whose result says done is skipped, and
# one that does not finish is retried up to 3 times (bfcl keeps every answer and generates only the missing ones).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
# one copy per VM: two copies append their answers to the same files, which bfcl then refuses to score
exec 9> /tmp/run_e18.lock
flock -n 9 || { echo "run_e18.sh is already running on this VM"; exit 0; }
MODEL="${MODEL:?set MODEL}"; EXAMS="${EXAMS:?set EXAMS}"
IFS='|' read -r name registry weights <<< "$MODEL"
PROJECT=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
BUCKET="${BUCKET:-gs://$PROJECT-alife}"
REPO="$PWD"; WORK="$HOME/e18"; mkdir -p logs "$WORK" "$HOME/models"
if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token --project "$PROJECT" 2>/dev/null) && export HF_TOKEN
fi
# venvs as in E14 (run_bfcl_arms.sh)
[ -f /usr/include/python3.10/Python.h ] || sudo apt-get install -yq python3.10-dev
# the newer vLLM compiles kernels at start-up, as on E14's VMs; a fresh VM has no C++ compiler
command -v g++ > /dev/null || { sudo apt-get -o DPkg::Lock::Timeout=600 update -q > /dev/null 2>&1
  sudo apt-get -o DPkg::Lock::Timeout=600 install -yq g++ > /dev/null 2>&1; }  # waits for the boot-time updates
if [ -n "${NEW_STACK:-}" ]; then
  VENV="$HOME/bfcl-new-venv"; VLLM="$HOME/vllm-new-venv/bin/vllm"
  [ -x "$VENV/bin/bfcl" ] || { uv venv --python 3.12 "$VENV"; uv pip install --python "$VENV/bin/python" "bfcl-eval==2026.3.23" transformers soundfile setuptools; }
  [ -x "$VLLM" ] || { uv venv --python 3.12 "$HOME/vllm-new-venv"; uv pip install --python "$HOME/vllm-new-venv/bin/python" "vllm>=0.11" ninja; }
else
  VENV="$HOME/bfcl-venv"; VLLM="$VENV/bin/vllm"
  [ -x "$VENV/bin/bfcl" ] || { uv venv --python 3.10 "$VENV"; uv pip install --python "$VENV/bin/python" "bfcl-eval[oss_eval_vllm]==2026.3.23" "transformers==4.51.3" soundfile setuptools; }
fi

# the e16 merge and the e17 embedder need torch and peft in the bfcl venv
"$VENV/bin/python" -c "import torch, peft" 2>/dev/null || uv pip install --python "$VENV/bin/python" torch "peft>=0.15"

# a downloaded model's weights, once (the e17 tokenizer and the e16 merge read them from here)
case "$weights" in
  hf:*) [ -f "$HOME/models/$name/config.json" ] || "$VENV/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "${weights#hf:}" "$HOME/models/$name" > /dev/null ;;
esac

done_in() {  # exam name: is its summary in the bucket done?
  gcloud storage cat "$BUCKET/bfcl/e18/$1.json" 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('stage') == 'done' else 1)" 2>/dev/null
}

for x in $EXAMS; do
  IFS='|' read -r exp run <<< "$x"
  exam="$exp-$name-$run"
  done_in "$exam" && { echo "$(date) $exam: done already"; continue; }
  w="$weights"; env_extra=()
  [ "$registry" != "${registry#e14-}" ] && env_extra+=(E18_REGISTER_E14=1)
  case "$exp" in
    e15)
      gcloud storage cp -q "$BUCKET/e15/$name/$run/notes-1200.txt" "$WORK/$exam-notes.txt" || { echo "$exam: no final notes"; continue; }
      env_extra+=(BFCL_CMD="python -m lab.e18.exam_bfcl" E18_NOTES="$WORK/$exam-notes.txt") ;;
    e17)
      gcloud storage cp -q "$BUCKET/e17/$name/$run/memory-1200.jsonl" "$WORK/$exam-memory.jsonl" || { echo "$exam: no final memory"; continue; }
      case "$weights" in hf:*) tok="$HOME/models/$name" ;; *) tok="" ;; esac
      case "$run" in *600*) budget=600 ;; *) budget=1200 ;; esac
      env_extra+=(BFCL_CMD="python -m lab.e18.exam_bfcl" E18_MEMORY="$WORK/$exam-memory.jsonl" E18_BUDGET="$budget" E18_TOKENIZER="${tok:-TOKENIZER_FROM_REGISTRY}") ;;
    e16|e19)
      mkdir -p "$WORK/$exam-adapter"
      gcloud storage cp -r -q "$BUCKET/$exp/$name/$run/adapter-1200/*" "$WORK/$exam-adapter/" \
        && [ -f "$WORK/$exam-adapter/adapter_model.safetensors" ] || { echo "$exam: no final adapter (nothing learnt: the floor)"; continue; } ;;
  esac
  if [ "$exp" = e17 ] && [ -z "$tok" ]; then  # bfcl's own model: its Hugging Face name
    tok="$("$VENV/bin/python" -c "import sys; from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING as M; print(M[sys.argv[1]].model_name)" "$registry")"
    env_extra=("${env_extra[@]/TOKENIZER_FROM_REGISTRY/$tok}")
  fi
  if [ "$exp" = e16 ] || [ "$exp" = e19 ]; then
    case "$weights" in
      hf:*) src="$HOME/models/$name"
            [ -f "$src/config.json" ] || "$VENV/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "${weights#hf:}" "$src" > /dev/null ;;
      *) src="$("$VENV/bin/python" -c "import sys; from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING as M; print(M[sys.argv[1]].model_name)" "$registry")" ;;
    esac
    out="$WORK/$exam-merged"
    [ -f "$out/config.json" ] || PYTHONPATH="$REPO" "$VENV/bin/python" -m lab.e18.merge --base "$src" --adapter "$WORK/$exam-adapter" --out "$out" >> "logs/e18-$exam.log" 2>&1
    [ -f "$out/config.json" ] || { echo "$(date) $exam: merge failed (logs/e18-$exam.log)"; continue; }
    w="local:$out"
  fi
  for try in 1 2 3; do
    done_in "$exam" && break
    [ -d "$HOME/bfcl-e18/$exam/result" ] && PYTHONPATH="$REPO" python3 -m lab.e18.dedupe "$HOME/bfcl-e18/$exam" >> "logs/e18-$exam.log" 2>&1
    env "${env_extra[@]}" RUN=e18 ARMS="$exam|$registry|$w" NO_SHUTDOWN=1 bash deploy/gcp/run_bfcl_arms.sh >> "logs/e18-$exam.log" 2>&1
  done
  done_in "$exam" && echo "$(date) $exam: done" || echo "$(date) $exam: not done after 3 tries"
done
[ -n "${NO_SHUTDOWN:-}" ] || sudo shutdown -h now

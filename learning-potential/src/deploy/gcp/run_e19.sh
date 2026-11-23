#!/bin/bash
# E19 on an A100 VM, from the repo folder: one LoRA run of E16's code from start to end, then its probes, then its
# BFCL exam, then the VM stops:
#   MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUN="teach-s16|teach|16" nohup bash deploy/gcp/run_e19.sh > logs/e19.log 2>&1 &
# MODEL as in run_e16.sh (NEW_STACK and MAX_LEN too); RUN is one run|arm|seed. Arms: E16's teach and judge, and
# teach-think (the reasoning kept, lab/e16/lora.py). Everything goes to $BUCKET/e19/<name>/<run>/ (training and probes,
# as E16) and $BUCKET/bfcl/e18/e19-<name>-<run>.json (the exam, as E18). Each step resumes where it stopped.
set -uo pipefail
RUN="${RUN:?set RUN}"; IFS='|' read -r run _ _ <<< "$RUN"
mkdir -p logs
echo "$(date) training $run"
EXP=e19 RUNS="$RUN" NO_SHUTDOWN=1 bash deploy/gcp/run_e16.sh
sleep 30; pkill -f "vllm serve" 2>/dev/null; sleep 30  # the training server's GPU memory back before the next step
echo "$(date) probes $run"
EXP=e19 RUNS="$run" NO_SHUTDOWN=1 bash deploy/gcp/run_e16_probe.sh
sleep 30; pkill -f "vllm serve" 2>/dev/null; sleep 30
echo "$(date) exam $run"
EXAMS="e19|$run" NO_SHUTDOWN=1 bash deploy/gcp/run_e18.sh
echo "$(date) all done"
[ -n "${NO_SHUTDOWN:-}" ] || sudo shutdown -h now

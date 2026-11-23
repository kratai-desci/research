#!/bin/bash
# E14 on an L4 VM, from the repo folder: one tiny model's floor, frozen: the full BFCL v4 exam (as E11) with the model's
# own BFCL handler. PRACTICE_SETS="v3 v4 v5" also runs those E13 practice sets afterwards (off by default: E14 is
# about BFCL only). One model per VM:
#   ARM="falcon3-1b|tiiuae/Falcon3-1B-Instruct-FC|" nohup bash deploy/gcp/run_e14_floor.sh > logs/e14-floor.log 2>&1 &
# ARM is name|bfcl model|weights as in run_bfcl_arms.sh. Results: $BUCKET/bfcl/e14floor/<name>.json (BFCL), with live
# progress in $BUCKET/bfcl/e14floor/<name>-progress.json every minute; $BUCKET/bfcl/e14floor-v3/<name>.json etc. for
# the practice sets when asked. HF_TOKEN, if set, reaches the model downloads (gated
# models such as Llama-3.2 and Gemma-3).
#
# Safe against errors, and resumable:
# - a stage whose summary in the bucket says done is skipped;
# - bfcl keeps every answer it has generated and, run again, generates only the missing ones;
# - a stage that does not finish is retried, up to 3 times;
# - the work folders are backed up to $BUCKET/bfcl/e14floor-work/<name>/ every 5 minutes and restored on start, so
#   running this again on any VM (after a crash or a lost VM) continues where it stopped.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
ARM="${ARM:?set ARM}"
PROJECT=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
BUCKET="${BUCKET:-gs://$PROJECT-alife}"
IFS='|' read -r name _ _ <<< "$ARM"
WORK="$BUCKET/bfcl/e14floor-work/$name"
mkdir -p logs
# gated models (Llama-3.2, Gemma-3): the Hugging Face token comes from Secret Manager (secret hf-token), never printed
if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token --project "$PROJECT" 2>/dev/null) && export HF_TOKEN
fi

done_in() {  # run: is this model's summary for that run already done?
  gcloud storage cat "$BUCKET/bfcl/$1/$name.json" 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('stage') == 'done' else 1)" 2>/dev/null
}

# restore earlier work (answers generated so far), then back it up every 5 minutes
for run in e14floor e14floor-v3 e14floor-v4 e14floor-v5; do
  [ -d "$HOME/bfcl-$run/$name" ] || { mkdir -p "$HOME/bfcl-$run"; gcloud storage cp -r -q "$WORK/$run/$name" "$HOME/bfcl-$run/" 2>/dev/null; }
done
( n=0
  while sleep 60; do
    n=$((n + 1))
    # live progress of the BFCL exam, for the dashboard
    [ -d "$HOME/bfcl-e14floor/$name/result" ] && PYTHONPATH="$PWD" "$HOME/bfcl-venv/bin/python" -m lab.bfcl_progress \
      "$HOME/bfcl-e14floor/$name" "$name" > "$HOME/e14floor-progress-$name.json" 2>/dev/null \
      && gcloud storage cp -q "$HOME/e14floor-progress-$name.json" "$BUCKET/bfcl/e14floor/$name-progress.json"
    [ $((n % 5)) -eq 0 ] || continue  # the work folders every 5 minutes
    for run in e14floor e14floor-v3 e14floor-v4 e14floor-v5; do
      [ -d "$HOME/bfcl-$run/$name/result" ] && gcloud storage rsync -r -q "$HOME/bfcl-$run/$name" "$WORK/$run/$name"
    done
  done ) &
backup=$!

stage() {  # run [practice set]: up to 3 tries until the bucket summary says done
  local run=$1 practice=${2:-}
  for try in 1 2 3; do
    done_in "$run" && return 0
    [ $try -gt 1 ] && echo "$(date) $run: not done, try $try" >> "logs/e14floor-$name.log"
    RUN="$run" PRACTICE="$practice" ARMS="$ARM" NO_SHUTDOWN=1 bash deploy/gcp/run_bfcl_arms.sh >> "logs/$run-$name.log" 2>&1
  done
  done_in "$run" || echo "$(date) $run: still not done after 3 tries" >> "logs/e14floor-$name.log"
}

stage e14floor
for v in ${PRACTICE_SETS:-}; do stage "e14floor-$v" "$BUCKET/e13/practice-$v"; done

kill $backup 2>/dev/null
for run in e14floor e14floor-v3 e14floor-v4 e14floor-v5; do
  [ -d "$HOME/bfcl-$run/$name/result" ] && gcloud storage rsync -r -q "$HOME/bfcl-$run/$name" "$WORK/$run/$name"
done
[ -n "${NO_SHUTDOWN:-}" ] || sudo shutdown -h now

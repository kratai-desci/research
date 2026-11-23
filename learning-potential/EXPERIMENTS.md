# Running the experiments

How each experiment in the paper was launched. The lab notebook of each experiment (`notes/e13.md` … `notes/e20.md`)
gives its design, its runs as they happened (machines, restarts, timings) and its results; this page is the short
recipe. `src/` is the code at the end of the experiments (commit `a5fccc1` of the authors' working repository).

Every run went on Google Cloud, one NVIDIA L4 (24 GB) or A100 per machine, Singapore (`asia-southeast1`) or Tokyo when
Singapore ran out. Nothing here needs Google Cloud except the bucket the scripts back up to and restore from, and
Gemini on Vertex AI, which wrote the practice tasks (E13).

## Setup

1. **A GPU machine** with `uv`, the Google Cloud CLI and a bucket (`BUCKET`, default `gs://<project>-alife`).
   `src/deploy/gcp/create_vm.sh` creates an L4 VM and the bucket (settings in `src/deploy/gcp/config.sh`). Copy `src/`
   to the machine; every script runs from that folder (`bash deploy/gcp/run_e16.sh`, `python -m lab.e16.lora`).
2. **Python environments.** Each launch script installs its own, once per machine:
   - the *old stack* (Qwen3-1.7B, Llama-3.2-1B, Falcon3-1B, Gemma-3-1B): Python 3.10, `bfcl-eval[oss_eval_vllm]==2026.3.23`
     with the vLLM it pins (0.8.5), `transformers==4.51.3`, and `torch` + `peft==0.15.2` for LoRA;
   - the *new stack* (`NEW_STACK=1`: LFM2.5-1.2B and Qwen3.5-0.8B, which need a newer vLLM): Python 3.12,
     `bfcl-eval==2026.3.23` and `vllm>=0.11` (which resolved to 0.33.0 or 0.34.0, the latest release when each machine was set up) in a separate environment, with `torch` + `peft`;
   - for retrieval, `transformers` loads the sentence encoder (`sentence-transformers/all-MiniLM-L6-v2`).
3. **Gated models** (Llama-3.2, Gemma-3) need a Hugging Face token in `HF_TOKEN`; the scripts read it from Secret
   Manager (secret `hf-token`) when it is not set, and never print it.
4. **The practice pool** (E13), three sets of 800 tasks in BFCL's own format, is in `data/pool/`. The scripts copy
   it to the machine from the bucket (`e13/practice-v3`, `-v4`, `-v5`); upload `data/pool/practice-v*` there first.

## Models

Each launch names its model as `name|BFCL model|weights`: the name used in results, the BFCL model whose handler
(prompt and call format) is used, and where the weights come from (empty: BFCL's own; `hf:<repo>`: downloaded).

| Model | `MODEL` | Stack |
| --- | --- | --- |
| Qwen3-1.7B | `qwen3-1.7b\|Qwen/Qwen3-1.7B-FC\|` | old |
| LFM2.5-1.2B | `lfm2.5-1.2b\|e14-lfm2.5-1.2b\|hf:LiquidAI/LFM2.5-1.2B-Instruct` | new |
| Qwen3.5-0.8B | `qwen3.5-0.8b\|e14-qwen3.5-0.8b\|hf:Qwen/Qwen3.5-0.8B` | new |
| Llama-3.2-1B | `llama3.2-1b\|meta-llama/Llama-3.2-1B-Instruct-FC\|` | old |
| Falcon3-1B | `falcon3-1b\|tiiuae/Falcon3-1B-Instruct-FC\|` | old |
| Gemma-3-1B | `gemma3-1b\|google/gemma-3-1b-it\|` | old |

`e14-lfm2.5-1.2b` and `e14-qwen3.5-0.8b` are the handlers added in `src/lab/e14/new_handlers.py`; BFCL v4 does not
list these two models.

## E13: the practice ground

```bash
python -m lab.e13.practice --n 800 --out <dir> --model gemini-3-flash-preview   # inside bfcl-eval's environment
```

Three generations of 800 (`practice-v3`, `-v4`, `-v5`). A task is kept only if its reference runs in BFCL's
simulators without error, passes one strict review, and a solver completes it as BFCL scores it. The split
(`data/split.json`) was made once from the pooled 2,400 tasks with seed 15 (`make_split` in `lab/e15/notes.py`).

## E14: frozen scores

```bash
ARM="falcon3-1b|tiiuae/Falcon3-1B-Instruct-FC|" bash deploy/gcp/run_e14_floor.sh
```

The BFCL v4 exam (every category except web search, which needs a paid key) for each model, frozen. Qwen3-1.7B's
frozen score is the earlier baseline run, `deploy/gcp/run_bfcl.sh` (E11). `run_bfcl_arms.sh` also has branches for
an earlier experiment's models (`toolrl-*`, E12), whose code is not part of this release; no run here uses them.

## E15, E16, E17: learning on the job

One script per method; `RUNS` is a list of `run|arm|seed[|tasks]`, and every run practises the 1,200 practice tasks
in the split's order (seed 15) or another order (seeds 16 and 17).

```bash
# notes (E15): one VM per model, all of its runs side by side; frozen runs do all 2,400 tasks
MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUNS="frozen-s15|frozen|15 teach-s15|teach|15|1200 judge-s15|judge|15|1200" \
  bash deploy/gcp/run_e15.sh
MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUNS="teach-s15 judge-s15" bash deploy/gcp/run_e15_probe.sh

# continual LoRA (E16): one run per VM; arms teach, judge
MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUNS="teach-s15|teach|15" TASKS=1200 bash deploy/gcp/run_e16.sh
MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUNS="teach-s15 judge-s15" bash deploy/gcp/run_e16_probe.sh

# retrieved experience (E17): one VM per model; a fifth field sets the token budget (default 1,200)
MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUNS="teach-s15|teach|15 judge-s15|judge|15 teach600-s15|teach|15||600" \
  bash deploy/gcp/run_e17.sh
MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUNS="teach-s15 judge-s15 teach600-s15" bash deploy/gcp/run_e17_probe.sh
```

Add `NEW_STACK=1` (and `MAX_LEN=<tokens>` where the model's context must be capped) for LFM2.5 and Qwen3.5. The probe
scripts score every saved snapshot (notes, adapter or memory, every 100 tasks) on the 200 probe tasks as it appears.
Every script is resumable: runs save their state after every task (LoRA: every update), back up to the bucket every
5 minutes, restore on start and restart up to five times, so running a script again on any machine continues where it
stopped. The settings of each method are in the paper (Table 2) and at the top of each module.

## E18: the final BFCL exam

```bash
MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" EXAMS="e15|teach-s15 e16|teach-s15 e17|teach600-s15" bash deploy/gcp/run_e18.sh
```

The final state (task 1,200) of each run on the same exam as the frozen score: notes or retrieved examples in every
question's prompt (`lab/e18/exam_bfcl.py`), LoRA adapters merged into the weights (`lab/e18/merge.py`).

## E19: seeds and the reasoning kept

```bash
MODEL="qwen3-1.7b|Qwen/Qwen3-1.7B-FC|" RUN="teach-think-s16|teach-think|16" bash deploy/gcp/run_e19.sh
```

One LoRA run from start to end (training, probes, exam), then the machine stops. Part A: seeds 16 and 17 for
teaching on Llama, Gemma, LFM2.5, Qwen3.5 and Qwen3-1.7B, and judging on Qwen3-1.7B. Part B: `teach-think` (teaching
with the reasoning kept) on Qwen3-1.7B, seeds 15–17. Runs went on A100s.

## E20: statistics and costs

```bash
python -m lab.e20.extract --runs <e15/ e16/ e17/ e19/ from the bucket> --exams <bfcl/e18> \
    --floors <bfcl/e14floor-work> --e11 <bfcl/e11> --out ../data/per_question
uv run --with numpy --with scipy python -m lab.e20.stats
python -m lab.e20.costs --runs <e15/ e16/ e17/ e19/ from the bucket>
```

No new runs. `extract` needs the raw run folders (bucket, or the archive once published); `stats` needs only
`data/per_question/`.

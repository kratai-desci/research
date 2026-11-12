# Running the experiments

How each experiment in the paper was launched. The settings below are read from the lives themselves:
every life stored its arguments, and `data/configs.csv` lists them, one row per life (extracted by
`src/extract_configs.py`). Where this page and `configs.csv` differ, `configs.csv` is what ran.

`src/lab/` is the code at the end of the experiments (commit `fa12906` of the authors' working repository). Some
options were added while the experiments ran (for example `--lesson-once` and `--min-lessons` before Experiment 3's
sleep variant); a life whose config lacks an option ran with its default, which reproduces the earlier behaviour.
Lives ran on Google Cloud, one NVIDIA L4 (24 GB) per machine: eight lives at a time on one machine for Experiment 1
and its pilot, two at a time for the rest. Nothing below needs Google Cloud except Gemini, which plays the realistic
user and τ²-bench's customer and graders.

## Setup

1. **Python environment** for `src/` (Python 3.12+): `torch`, `transformers`, `peft`, `vllm`, `requests`
   (the night's training and the sense run on PyTorch; the creature is served by vLLM).
2. **WebShop** (Experiments 1-4), the small product set (1,000 products, 13 human-written goals) in its own Python
   3.8 environment, then its two goal sets behind a small HTTP bridge:
   ```bash
   bash src/lab/scripts/setup_webshop.sh                      # into vendor/WebShop
   cd vendor/WebShop
   .venv/bin/python ../../src/lab/scripts/webshop_server.py synthetic &   # port 3100: generated goals
   .venv/bin/python ../../src/lab/scripts/webshop_server.py human &       # port 3101: human-written goals
   ```
3. **The creature** served by vLLM with runtime LoRA loading (each morning loads the previous night's adapter):
   ```bash
   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve Qwen/Qwen3-0.6B --port 8001 --enable-lora \
     --max-loras 8 --max-lora-rank 64 --max-model-len 8192 --seed 0
   ```
   For τ²-bench add `--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384`, and serve the
   sense on a second port: `vllm serve Qwen/Qwen3-0.6B --port 8002 --enable-lora --max-lora-rank 64 --enforce-eager`.
4. **The sense** (judge `sense` only): copy the adapter that experiment used into `lab/data/sense/adapter`
   (WebShop) or `lab/data/sense3/adapter` (τ²-bench), with the fixed reactions of the daily check in
   `lab/data/sense/probe.jsonl`. The adapters and the reactions are in the level-3 archive (see README).
   The reader picks the version from the adapter's `calibration.json`: none = v1 (fixed 70% threshold),
   a temperature and threshold = v2, `"version": 3` = v3's two gates.
5. **Gemini** (Experiments 3-5): Vertex AI credentials and `GOOGLE_CLOUD_PROJECT` (WebShop's realistic user) or
   `VERTEXAI_PROJECT` with `VERTEXAI_LOCATION=global` (τ²-bench), and model `gemini-3-flash-preview`.
6. **τ²-bench** (Experiment 5): `bash src/setup/tau2_gemini.sh` clones τ²-bench at commit `033bf08` and switches
   its graders to Gemini. Set `TAU2_DIR` to that folder.

Run every command from the folder that holds `lab/` (here, `src/`), with `--backend vllm`.

## WebShop (Experiments 1-4): `python -m lab.life`

Common to every WebShop life: `--backend vllm --model Qwen/Qwen3-0.6B --practice 30 --probes 50 --human-probes 13
--threshold 0.5 --temperature 0.7 --seed <0-4> --experiment <lab name>`. The judge is `--signal`, with the lab's
names: `user` is the paper's *stated verdict*.

| Paper | Lab name | Judges (`--signal`) | Seeds | Other arguments | Sense adapter |
| --- | --- | --- | --- | --- | --- |
| Exp 1, pilot | `growing-from-use` | oracle, user, self, both, none | 0 | `--days 6` | - |
| Exp 1 | `growing-from-use-14d` | oracle, user, self, both, none | 0-4 | `--days 14` | - |
| Exp 1, year | `lifespan-365d` | both (seeds 0-4), oracle (seed 0) | | `--days 365 --exam-every 1:30,2:90,7` | - |
| Exp 1, adapters | `tree-rings-14d` | both | 0-4 | `--days 14 --brain <constant, lr-decay, rings-equal, rings-tree> --condition <same>` | - |
| Exp 2 | `feedback-sense-14d` | oracle, both, sense | 0-4 | `--days 14` | sense **v1** (see below) |
| Exp 3 | `realistic-user-14d` | oracle, both, sense | 0-4 | `--days 14 --user llm` | sense v2 |
| Exp 3, sleep variant | `lesson-once-14d` | oracle, both, sense | 0-4 | `--days 14 --user llm --lesson-once --min-lessons 5` | sense v2 |
| Exp 4 | `clear-verdict-14d` | oracle, both, sense | 0-4 | `--days 60 --stream-days 14 --exam-every 1:30,2 --user llm --lesson-once` | sense v3 |

Notes:

- **The sense in Experiment 2.** Experiment 2 started (3 November 2026) before sense v2 existed. Its lives used the
  first sense build (`sense1-adapter` in the archive): Qwen3-0.6B with a rank-16 LoRA trained on 45,514 WildFB
  reactions with WildFB's own labels, no calibration file, so the reader's fixed 70% threshold. Its daily check on
  the fixed reactions differs from v2's (it agrees with Experiment 3's sense on 213 of 302 reactions on day 0).
- **Experiment 4** ran 14 days with daily exams, then every life was resumed (`--resume`) to day 60, examined daily
  to day 30 and every second day after. `--stream-days 14` keeps the practice tasks the 14-day life would have had.
- **The year** examined daily to day 30, every second day to day 90, then weekly.
- Lives with the same seed meet the same practice tasks in the same order, and the user's random draws come from
  the same seed; practice sampling is not seeded.

## τ²-bench (Experiment 5): `python -m lab.life_tau2`

```bash
SLEEP_MAX_SEQ=10240 SLEEP_BATCH=1 TAU2_DIR=~/tau2-bench \
  python -m lab.life_tau2 --model Qwen/Qwen3-<0.6B|1.7B> --signal <oracle|sense> --seed <0-2> --days 30 \
    --experiment tau2-learning-30d --condition <signal>-<size>
```

The defaults are what ran: 30 practice conversations a day on the retail and airline training tasks
(`--domains retail,airline`), exams on day 0 and every 3 days on the retail, airline and telecom test tasks
(`--exam-domains retail,airline,telecom --exam-every 3`), practice at temperature 0.7 for the agent and the
customer, exams at 0. Sense v3 is read from `lab/data/sense3/adapter`. Each lesson counts once in sleep (built in;
τ²-bench lives have no `--lesson-once` option) and there is no retry.

## Building the senses

Each step is a module of `lab.sense` (see the docstrings): `sources` gathers the reactions from the six public
datasets (not redistributed here; Table A3 gives sources and licences), `teach` / `teach3` score them with the
Qwen3-32B teacher (AWQ, on vLLM), `gaps` has the teacher write the hard reactions, `train2` / `train3` train and
calibrate the v2 and v3 students (`train2 --hold-out <task type>` for the leave-one-task-out builds). Their status
files, with every number the paper reports, are in `data/sense/`.

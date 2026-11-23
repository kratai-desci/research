# Learning potential of small language models on the job

*Can a small language model learn a new tool skill while it works, and by which route: its own notes, retrieved
worked examples, or its weights?*

Six small open models (0.8B–1.7B: Qwen3-1.7B, LFM2.5-1.2B, Qwen3.5-0.8B, Llama-3.2-1B, Falcon3-1B, Gemma-3-1B) each
work through 1,200 multi-turn tool-use tasks built on the Berkeley Function Calling Leaderboard (BFCL v4) tool
simulators, and learn from each task by one of three on-device methods: self-written notes, retrieval of past worked
examples, or continual low-rank adaptation (LoRA). Feedback is either *teaching* (the reference answer is shown) or
*judging* (only pass or fail). Learning is tested on 200 unseen tasks every 100 tasks and, at the end, on the
BFCL v4 exam (every category except web search, which needs a paid search service).

| Folder | What it holds |
| --- | --- |
| `paper/` | the paper in LaTeX, in Elsevier's CAS two-column format for *Knowledge-Based Systems*: `main.tex`, `refs.bib`, `highlights.txt`, `figures/` (with `make_figures.py`), `thumbnails/` (the small icons the template prints beside the email address) |
| `src/` | the code the results come from (`lab/`) and the scripts that launched every run on Google Cloud (`deploy/gcp/`) |
| `data/` | the fixed task split, the per-question result of every run, the statistical tests and the costs, from which every number, test and figure in the paper can be recomputed |
| `notes/` | the lab notebook of each experiment (E13–E20): design, runs, results and what was learned, as written while the work ran (see the note below) |
| `EXPERIMENTS.md` | how each experiment was set up and launched |

The notes keep the lab's working terms: their "floor" is the paper's frozen score, and their "zero-floor models" are
the paper's models below the floor. They were written as the work ran and are kept as a record; where a note and the
paper differ, the paper and `data/` are the checked version.

## Funding and licence

This work was funded by Decentralized Science, Inc., which owns the intellectual property in it. The code in `src/`
is under the MIT licence (`LICENSE`); the data, notes, paper and figures are under CC BY 4.0 (`LICENSE-DATA`).
Copyright is held by Decentralized Science, Inc. Third-party benchmarks (BFCL), models and libraries keep their own
licences. One file is third-party data and is not under either licence: `src/lab/e13/personas.jsonl` holds 999
personas from PersonaHub (`proj-persona/PersonaHub`, persona configuration, first 1,000 rows, unchanged), under CC
BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/); the third generation of practice tasks gave each
task's user one of these personas, as BFCL did for its own multi-turn questions.

## Rebuild the numbers and figures

```bash
cd src && uv run --with numpy --with scipy python -m lab.e20.stats && cd ..     # every test -> data/stats.json
uv run --with matplotlib python paper/figures/make_figures.py                   # Figures 3-8 -> paper/figures/, and data/side_effects.json
```

Both run in seconds on a laptop and reproduce `data/stats.json` and `data/side_effects.json` byte for byte.
Compile the paper in `paper/` with `tectonic main.tex` (or `latexmk -pdf main.tex`), or upload `paper/` to Overleaf.
Figure 1 (the study design) is drawn in TikZ inside `main.tex`.

## Data (`data/`)

| File | What it holds |
| --- | --- |
| `pool/` | the 2,400 practice tasks with their reference answers (see below) |
| `split.json` | the fixed split of the 2,400-task pool (seed 15): the 1,200 practice tasks in practice order, the 1,200 held out, and the 200 probe tasks among them |
| `per_question/bfcl.json` | every BFCL v4 question in the 20 categories examined (all but web search), solved (1) or not (0), for each model's frozen score and every final-state exam (52 exams) |
| `per_question/probe.json` | the 200 probe tasks, for each model's start and every probe snapshot (every 100 tasks) of every run (630 snapshots) |
| `per_question/practice.json` | the 1,200 practice tasks, for every frozen and learning run (58 runs); runs with seeds 16 and 17 practised the same tasks in another order and are aligned by task |
| `stats.json` | every paired test in the paper (McNemar exact, paired bootstrap 95% intervals, sign-flip over three seeds, Holm within families), from `src/lab/e20/stats.py` |
| `side_effects.json` | the calling-versus-abstaining split and memory change of every run (BFCL), with tests (Holm across the grid); behind Fig. 6; written by `paper/figures/make_figures.py` |
| `costs.json` | what each run cost as it ran: GPU, runs sharing it, seconds per task, hours, size of the saved state and prompt tokens added (Section 4.6), from `src/lab/e20/costs.py` |
| `results/e15.json` … `e19.json` | each experiment's run summaries as the lab recorded them (practice curves, gain over frozen, probe curves, exams) |

The per-question tables are a fixed task order plus one string of `0`/`1` per run (`-` = not attempted). They were
extracted once from the raw run folders by `src/lab/e20/extract.py`, which checks each BFCL category's count against
BFCL's own score file.

The full step-by-step logs, notes snapshots, retrieval memories and LoRA adapters are too large for git and will be
archived separately (link to follow).

### The practice pool (`data/pool/`)

The 2,400 practice tasks, in three generations of 800 (`practice-v3`, `practice-v4`, `practice-v5`), each in BFCL's
own format so that BFCL's runner and checker read them unchanged:

| File | What it holds |
| --- | --- |
| `BFCL_v4_multi_turn_base.json` | the tasks (one JSON per line, ids `multi_turn_base_<i>`): the tool classes, the simulators' initial state and the user turns |
| `possible_answer/BFCL_v4_multi_turn_base.json` | the hidden reference calls of each turn (`ground_truth`) |
| `tasks.jsonl` | the generator's record of each kept task: its turns with their reference calls |
| `rejected.jsonl` | the tasks the generator dropped, with the reason (failed execution, review or solver) |

`data/split.json` names its tasks as `[set, id]`. Its practice and held-out lists are rebuilt exactly from the pool
by `make_split` in `src/lab/e15/notes.py` (seed 15); the 200 probe tasks (67, 67 and 66 from the three generations)
were drawn once from the held-out tasks and are fixed in the file. The initial states contain made-up user names and
passwords for BFCL's simulated services; none is real.

## Code (`src/`)

`src/lab/` is a copy of the experiment code at commit `a5fccc1` of the authors' working repository. The only changes
are the paths of the data files, which now point at `data/` (`split.json` in `e15/notes.py`, `e17/nearcopy.py` and
`e20/extract.py`; the outputs of `e20/stats.py` and `e20/costs.py`), and the experiment bucket, which `e20/costs.py`
now reads from the `BUCKET` environment variable.

| Module | What it does |
| --- | --- |
| `lab/e13/` | the practice ground: an LLM (Gemini 3 Flash on Vertex AI, `lab/vertex.py`) writes multi-turn tasks for BFCL's tool simulators (`practice.py`); `practice_bfcl.py` runs BFCL's own runner and checker on them |
| `lab/e14/new_handlers.py` | BFCL handlers for the models BFCL v4 does not list (LFM2.5, Qwen3.5) |
| `lab/e15/` | the loop and the notes method (`notes.py`), the probe (`probe.py`) |
| `lab/e16/lora.py` | continual LoRA: teaching, judging and teaching with the reasoning kept (`teach-think`) |
| `lab/e17/` | retrieved experience (`rag.py`), and the near-copy check of probe against practice tasks (`nearcopy.py`) |
| `lab/e18/` | the final BFCL exam: notes or retrieved examples in the prompt (`exam_bfcl.py`), adapters merged into the weights (`merge.py`) |
| `lab/e20/` | per-question extraction, statistics and costs |
| `lab/bfcl_summary.py`, `lab/bfcl_progress.py` | summaries and live progress of a BFCL run |
| `deploy/gcp/` | one launch script per experiment (see `EXPERIMENTS.md`) |

## Experiment names

The paper describes one study; the code, data and notes keep the lab's experiment names.

| Paper | Lab name |
| --- | --- |
| Practice ground (Section 3.1) | E13 |
| Frozen scores on BFCL v4 (Section 3.2, Table 1) | E14 (Qwen3-1.7B's from the earlier BFCL baseline, E11) |
| Self-written notes (Sections 3.4.1, 4.1) | E15 |
| Continual LoRA (Sections 3.4.3, 4.3) | E16, plus seeds 16 and 17 from E19 |
| Retrieved experience (Sections 3.4.2, 4.2) | E17 |
| Final BFCL exam (Section 3.6) | E18 |
| Teaching with the reasoning kept (Sections 3.4.4, 4.4) | E19 |
| Statistical analysis and costs (Sections 3.7, 4.6) | E20 |

In the code, *teaching* is the arm `teach`, *judging* is `judge`, and teaching with the reasoning kept is `teach-think`.

## Figures

| Paper | File (`paper/figures/`) | Drawn by |
| --- | --- | --- |
| Figure 1 | (TikZ in `main.tex`) | `main.tex` |
| Figure 2 | (a practice task, in `main.tex`) | `main.tex` |
| Figure 3 | `notes_curves` | `make_figures.py` |
| Figure 4 | `retrieval_curves` | `make_figures.py` |
| Figure 5 | `lora_curves` | `make_figures.py` |
| Figure 6 | `overcalling` | `make_figures.py` |
| Figure 7 | `reasoning_curves` | `make_figures.py` |
| Figure 8 | `transfer` | `make_figures.py` |
| (not in the paper) | `side_effects`: the full grid behind Fig. 6, every run and BFCL group | `make_figures.py` |

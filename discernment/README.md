# Learning through Discernment

*When a small language model learns from user feedback on the job, and when it decays.*

A small language model (Qwen3 0.6B or 1.7B) works by day, and each night a judge decides which of its attempts
become lessons for a small LoRA adapter. We compare judges over lives of 14 days to a year on WebShop and
τ²-bench: the true outcome (oracle), the user's stated verdict, the model's own reflection (self), its reflection
while hearing the user (both), and a separate, frozen "feedback sense" that reads the user's reaction.

| Folder | What it holds |
| --- | --- |
| `paper/` | the paper in LaTeX: `main.tex`, one file per section in `sections/`, `references.bib`, `figures/` |
| `src/` | the code the results come from (`lab/`), and the scripts that rebuild the data, statistics and figures |
| `data/` | per-life records from which every number, test and figure in the paper can be recomputed |
| `EXPERIMENTS.md` | how each experiment was set up and launched, with the settings every life ran with |

## Rebuild the numbers and figures

```bash
uv run --with scipy python src/stats.py           # every test in Appendix B -> data/stats.json
uv run --with matplotlib python src/figures.py    # the data figures -> paper/figures/
uv run --with matplotlib python src/diagrams.py   # Fig. 1 and the sense diagram -> paper/figures/
```

Compile the paper in `paper/` with `tectonic -X compile main.tex` or `latexmk -pdf main.tex` (pdflatex + bibtex), or upload `paper/` to Overleaf.

## Data (`data/`)

| File | One row per | Columns |
| --- | --- | --- |
| `lives.csv` | life (one seed) | `life`, `paper` (where it appears, e.g. "Exp 3"), `lab_experiment`, `judge`, `seed`, `model`, `status` |
| `exams.csv` | life, day and exam split (τ²-bench: also domain) | `split` (`probe` = the held-out exam; `probe-human` = WebShop's 13 human-written goals; `probe-transfer` = τ²-bench telecom), `tasks`, `errors`, `mean_score` (0-100), `full_successes` |
| `attempts.csv.gz` | practice attempt or retry | `day`, `kind` (practice / retry), `task`, `score` (reward × 100), `lesson` (yes / no / none: whether the judge made it a lesson), `verdict` (the judge's own words or the sense's label and confidence), `request`, `user_said` (the simulated user's reaction; WebShop prefixes the behaviour and next move) |
| `sense_checks.csv.gz` | daily check of the sense on fixed recorded reactions | `kind`, `item`, `correct`, `said`, `sense_read` |
| `configs.csv` | life | the settings it ran with (judge, days, exam schedule, user, sleep options, adapter design, sense), with start and finish times; see `EXPERIMENTS.md` |
| `sense/` | the two sense builds | teacher and training status for v2 and v3, the nine leave-one-task-out builds of v2, and the 2,622 recorded reactions of the v2/v3 test (`sense3_live.jsonl`) |

## Full logs and models (level 3)

The full step-by-step logs and the three sense adapters are too large for git and are archived separately
(link to follow): `discernment-lives.db.zst`, one SQLite file (2.7 GB unpacked, zstd-compressed) holding, for each
of the 127 lives, every attempt (practice, retry and exam), every step of it (the model's thought, action and
input, and the page or reply it got back), the judge's verdicts and the sense's daily checks; and `sense/`, the
v1, v2 and v3 sense adapters (LoRA on Qwen3-0.6B) with the 302 fixed reactions of the daily check.
`src/extract_archive.py` builds the database from the experiment databases; its docstring describes the tables.

The sense was trained on labels of public datasets (WildFB, WildFeedback/WildChat, Taskmaster, MultiWOZ 2.2,
ReDial, CCPE-M); those datasets are not redistributed here. See the paper's Table A3 for sources and licences.

## Code (`src/`)

`src/lab/` is a copy of the experiment code at commit `fa12906` of the authors' working repository: the creature
and its judges (`creature.py`, `agent.py`), a life on WebShop and on τ²-bench (`life.py`, `life_tau2.py`), the
night's training (`sleep.py`, `sleep_torch.py`), the simulated users (`user.py`, `user_llm.py`), the benchmark
bridges (`tracks/`) and the feedback sense (`sense/`: data, teacher, student training and the reader used during a
life). `src/extract_data.py` and `src/extract_configs.py` produced `data/` from the experiment databases. `src/lab/scripts/`
sets up WebShop and its HTTP bridge, and `src/setup/tau2_gemini.sh` sets up τ²-bench as Experiment 5 ran it.

## Experiment names

The paper numbers experiments in story order; the code and data use lab names.

| Paper | Lab name |
| --- | --- |
| Experiment 1 (five judges; pilot; year of life; adapter designs) | growing-from-use-14d; growing-from-use; lifespan-365d; tree-rings-14d |
| Experiment 2 (a separate, frozen sense; the first build, v1) | feedback-sense-14d |
| Experiment 3 (realistic users; sleep variant) | realistic-user-14d; lesson-once-14d |
| Experiment 4 (sense v3, 60 days) | clear-verdict-14d |
| Experiment 5 (τ²-bench) | tau2-learning-30d |

The judge called *stated verdict* in the paper is `user` in the code.

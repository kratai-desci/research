"""E20: what each learning method costs, from the E15 to E19 runs as they ran (status files, logs and the sizes of the
saved state in the bucket). Nothing is rerun, so the hardware is what each run had:

  E15 notes      one L4 per model, its three runs (frozen, teaching, judging) side by side on one vLLM server
  E16 LoRA       one GPU per run at the end (A100s, some L4s before the spot VMs were lost), so times are mixed
  E17 retrieval  one L4 per model, its runs side by side; Qwen3-1.7B on one A100 (three runs side by side)
  E19 LoRA       one A100 per run: the cleanest LoRA timings

Per run: seconds per practice task (solving plus learning), the share of that spent learning (LoRA: training and
loading the adapter, from the log), the learned state at task 1,200 and its size, how the state grows, and what it adds
to every prompt afterwards.

    python -m lab.e20.costs --runs <folder with e15/ e16/ e17/ e19/ as in the bucket> [--out ../data/costs.json]
Sizes come from the bucket (gcloud storage ls -l).
"""
import argparse
import os
import json
import subprocess
from pathlib import Path

BUCKET = os.environ.get("BUCKET", "gs://<project>-alife")  # the experiment bucket
MODELS = ["qwen3-1.7b", "lfm2.5-1.2b", "qwen3.5-0.8b", "llama3.2-1b", "falcon3-1b", "gemma3-1b"]
HARDWARE = {  # (experiment, model) -> GPU and how many runs shared it
    **{("e15", k): ("L4", 3) for k in MODELS},
    **{("e17", k): ("L4", 2) for k in MODELS}, ("e17", "qwen3-1.7b"): ("A100", 3),
    **{("e16", k): ("mixed (A100 or L4)", 1) for k in MODELS},
    **{("e19", k): ("A100", 1) for k in MODELS},
}
METHOD = {"e15": "notes", "e16": "LoRA", "e17": "retrieval", "e19": "LoRA"}


def sizes() -> dict[str, int]:
    pats = [f"{BUCKET}/e15/*/*/notes-1200.txt", f"{BUCKET}/e17/*/*/memory-1200.jsonl",
            f"{BUCKET}/e16/*/*/adapter-1200/adapter_model.safetensors",
            f"{BUCKET}/e19/*/*/adapter-1200/adapter_model.safetensors"]
    out = {}
    for p in pats:
        r = subprocess.run(["gcloud", "storage", "ls", "-l", p], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[-1].startswith("gs://"):
                rel = parts[-1][len(BUCKET) + 1:].split("/")
                out["/".join(rel[:3])] = int(parts[0])
    return out


def lines(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for l in p.read_text().splitlines():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "data" / "costs.json"))
    a = ap.parse_args()
    runs, size = Path(a.runs), sizes()
    rows = []
    for exp in ("e15", "e16", "e17", "e19"):
        for key in MODELS:
            for d in sorted((runs / exp / key).glob("*")) if (runs / exp / key).is_dir() else []:
                st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else None
                if not st or st.get("stage") != "done" or d.name.startswith(("pilot", "frozen", "check")):
                    continue
                log = lines(d / "log.jsonl")
                total = sum(r.get("secs", 0) for r in log)
                gpu, shared = HARDWARE[(exp, key)]
                row = {"run": f"{exp}/{key}/{d.name}", "method": METHOD[exp], "model": key, "arm": d.name,
                       "gpu": gpu, "runs_sharing_gpu": shared, "secs_per_task": st.get("secs_per_task"),
                       "hours": round(total / 3600, 1)}
                k = f"{exp}/{key}/{d.name}"
                if exp == "e15":
                    row.update({"state": "notes (text in every prompt)", "state_bytes": size.get(k),
                                "state_grows": "no: kept under 600 tokens by curation",
                                "prompt_tokens_added": st.get("notes_tokens"), "learning_share": None})
                elif exp == "e17":
                    row.update({"state": "log of solved tasks (text), retrieved into the prompt", "state_bytes": size.get(k),
                                "state_grows": "yes: one entry per logged task",
                                "prompt_tokens_added": f"up to {st['settings'].get('budget')}",
                                "examples_per_task": st.get("examples_per_task"), "learning_share": None})
                else:
                    upd = sum((r.get("update") or {}).get("secs", 0) for r in log)
                    b = size.get(k)
                    row.update({"state": "LoRA adapter (weights; merged, nothing added to the prompt)",
                                "state_bytes": b, "adapter_params": b // 4 if b else None,  # float32 weights
                                "state_grows": "no: fixed size", "prompt_tokens_added": 0,
                                "updates": st.get("updates"),
                                "learning_share": round(upd / total, 2) if total else None})
                rows.append(row)
    Path(a.out).write_text(json.dumps({"about": __doc__.split("\n\n")[0], "hardware": {
        f"{e}/{k}": {"gpu": g, "runs_sharing_gpu": n} for (e, k), (g, n) in HARDWARE.items()}, "runs": rows},
        indent=1) + "\n")
    print(f"{len(rows)} runs")


if __name__ == "__main__":
    main()

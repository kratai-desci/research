"""E15 and E16 probe: a run's saved notes (E15) or LoRA adapter (E16, --adapter), scored on the fixed probe tasks (200 held-out tasks never practised on), with
learning off. Each snapshot (notes-<n>.txt, saved every 100 tasks) is the run's whole learned state at that point, so
scoring it afterwards gives what the run would have scored then. No notes = checkpoint 0 (the model as it is).

    python -m lab.e15.probe --registry Qwen/Qwen3-1.7B-FC --data <v3> <v4> <v5> --notes notes-300.txt --out probe-300.json

Same handler, prompt and scoring as lab/e15/notes.py; the tasks run in parallel, since nothing carries over.
"""
import argparse
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from lab.e15.notes import SPLIT, Runner, load_pool


def _with(q: dict, system: str) -> dict:
    """The task with E17's retrieved examples as a system message at its start (unchanged when there are none)."""
    if not system:
        return q
    q = copy.deepcopy(q)
    q["question"][0].insert(0, {"role": "system", "content": system})
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--notes", default=None, help="a notes snapshot; none = no notes (checkpoint 0)")
    ap.add_argument("--adapter", default=None, help="E16: a LoRA adapter snapshot folder, loaded into vLLM (which must "
                                                    "serve with LoRA and runtime loading on)")
    ap.add_argument("--memory", default=None, help="E17: a memory snapshot (memory-<n>.jsonl); the examples for each "
                                                   "task are retrieved from it as in the run")
    ap.add_argument("--k", type=int, default=3, help="E17: examples shown at most")
    ap.add_argument("--budget", type=int, default=1200, help="E17: tokens for the examples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--set", default="probe", help="which tasks of the split: probe, or heldout-rest (held-out minus probe)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    sp = json.loads(Path(a.split).read_text())
    probe = {tuple(k) for k in sp["probe"]}
    keys = sp["probe"] if a.set == "probe" else [k for k in sp["heldout"] if tuple(k) not in probe]
    pool = {(src, q["id"]): (src, q, gt) for src, q, gt in load_pool(a.data, sp["seed"], split=None)}
    tasks = [pool[tuple(k)] for k in keys]
    notes = [l for l in Path(a.notes).read_text().splitlines() if l.strip()] if a.notes else []

    run = Runner(SimpleNamespace(registry=a.registry, model_path=a.model_path, temp=0.7, seed=sp["seed"]))
    systems = {}
    if a.memory:  # E17: retrieve every task's examples first (the embedder runs one task at a time)
        from lab.e17.rag import Embedder, Memory
        mem = Memory(Embedder())
        mem.load(Path(a.memory))
        for j, (_, q, _) in enumerate(tasks):
            systems[j] = mem.retrieve(q, a.k, a.budget, run.ntok)[0]
    lora = None
    if a.adapter and (Path(a.adapter) / "adapter_model.safetensors").exists():
        import requests
        lora = f"probe-{Path(a.adapter).name}"
        root = run.handler.base_url.rstrip("/")
        requests.post(f"{root}/unload_lora_adapter", json={"lora_name": lora}, timeout=600)  # a leftover of a crash
        r = requests.post(f"{root}/load_lora_adapter", json={"lora_name": lora, "lora_path": str(Path(a.adapter).resolve())},
                          timeout=600)
        r.raise_for_status()
        run.handler.model_path_or_id = lora
    t0 = time.time()
    # ids far from the practice runs' so the tool simulators' instances never collide
    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(lambda t: run.solve(900_000 + t[0], _with(t[1][1], systems.get(t[0], "")), t[1][2], notes),
                          enumerate(tasks)))
    if lora:
        requests.post(f"{root}/unload_lora_adapter", json={"lora_name": lora}, timeout=600)
    rows = [{"set": src, "id": q["id"], "valid": s["valid"], "error": (s["error"] or {}).get("error_type")}
            for (src, q, _), s in zip(tasks, out)]
    solved = sum(r["valid"] for r in rows)
    res = {"registry": a.registry, "notes": a.notes, "adapter": a.adapter, "memory": a.memory, "n_notes": len(notes), "set": a.set, "total": len(rows),
           "solved": solved, "score": round(100 * solved / len(rows), 2), "secs": round(time.time() - t0, 1),
           "results": rows}
    tmp = Path(a.out + ".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(a.out)
    print(f"{a.memory or a.adapter or a.notes or 'no notes'}: {solved}/{len(rows)} ({res['score']}%) in {res['secs']} s", flush=True)


if __name__ == "__main__":
    main()

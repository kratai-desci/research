"""A life on τ²-bench (E10): serve simulated customers by day, sleep at night, exam every few days.

The creature (Qwen3 with its learning LoRA, on vLLM) is τ²-bench's agent; Gemini plays the customer. Each day:
  1. The day's training tasks (retail and airline training splits by default), with some randomness. A proxy
     in front of vLLM records every turn the agent takes.
  2. Each conversation is judged: oracle (τ²-bench's true outcome) or sense (the frozen sense v3 reading how the
     conversation ended: the customer's last real words). A conversation judged good is a lesson: all its agent
     turns, exactly as the agent saw them; each lesson counts once in sleep, however many turns it had.
  3. Sleep on the lessons (replaying recent ones).
  4. Every --exam-every days (and on day 0): the test splits of all exam domains, carefully, never learned from.
Unlike the WebShop life, there is no retry after a bad verdict: τ²-bench's customer cannot be asked to react twice.

  uv run python -m lab.life_tau2 --model Qwen/Qwen3-1.7B --signal sense --seed 0 --days 30 --name e10-1.7B-sense-s0
"""

import argparse
import json
import time
from pathlib import Path

from lab import sleep, store
from lab.tracks import tau2

LIVES_DIR = Path(__file__).parent / "data" / "lives"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8001", help="the creature's vLLM (base model, LoRAs loaded live)")
    p.add_argument("--sense-url", default="http://127.0.0.1:8002", help="the sense's vLLM (Qwen3-0.6B with the sense LoRA)")
    p.add_argument("--sense-adapter", default=str(Path(__file__).parent / "data" / "sense3" / "adapter"))
    p.add_argument("--signal", default="oracle", choices=["oracle", "sense"])
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--practice", type=int, default=30, help="training tasks per day")
    p.add_argument("--domains", default="retail,airline", help="domains learned on")
    p.add_argument("--exam-domains", default="retail,airline,telecom", help="domains examined (test splits)")
    p.add_argument("--exam-every", type=int, default=3)
    p.add_argument("--exam-limit", type=int, default=0, help="examine only the first N test tasks of each domain (a test run)")
    p.add_argument("--temperature", type=float, default=0.7, help="the agent's sampling temperature at work")
    p.add_argument("--proxy-port", type=int, default=8100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="")
    p.add_argument("--experiment", default="")
    p.add_argument("--condition", default="")
    p.add_argument("--resume", action="store_true")
    args, _ = p.parse_known_args()

    from transformers import AutoTokenizer

    from lab.models import VLLMModel

    size = args.model.split("-")[-1]
    name = args.name or f"tau2-{size}-{args.signal}-s{args.seed}"
    life_dir = LIVES_DIR / name
    state_file = life_dir / "state.json"
    state = json.loads(state_file.read_text()) if args.resume and state_file.exists() else None
    if not state:
        sleep.clear(life_dir)
        life_dir.mkdir(parents=True)
    domains, exam_domains = args.domains.split(","), args.exam_domains.split(",")
    exam_days = sorted({0, args.days, *range(args.exam_every, args.days + 1, args.exam_every)})
    exam_ids = {d: tau2.split(d, "test")[: args.exam_limit or None] for d in exam_domains}
    exam_size = sum(len(v) for v in exam_ids.values())
    db = store.connect()
    if state:
        run_id = state["run_id"]
        db.execute("DELETE FROM episodes WHERE run_id = ? AND day > ?", (run_id, state["day"]))
        db.commit()
        store.update(db, "runs", run_id, status="running", finished=None)
    else:
        run_id = store.insert(db, "runs", name=name, system="creature", model=args.model, track="tau2-life",
                              condition=args.condition or f"{args.signal}-{size}", status="running", experiment=args.experiment or None,
                              total=exam_size * len(exam_days) + args.practice * args.days, started=store.now(), config=vars(args))

    model = VLLMModel(args.model, base_url=args.vllm_url)
    tok = AutoTokenizer.from_pretrained(args.model)
    recorder = tau2.Recorder(args.vllm_url, args.proxy_port)
    reader = None
    if args.signal == "sense":
        from lab.sense.reader import SenseReader

        r = model.http.post(f"{args.sense_url}/v1/load_lora_adapter", json={"lora_name": "sense", "lora_path": args.sense_adapter}, timeout=300)
        if r.status_code != 200 and "already" not in r.text:
            raise RuntimeError(f"vLLM could not load the sense: {r.status_code} {r.text}")
        reader = SenseReader(args.sense_url, "sense", args.sense_adapter)

    adapter, memories, done, first = None, [], 0, 1
    if state:
        first, adapter, memories, done = state["day"] + 1, state["adapter"] and Path(state["adapter"]), state["memories"], state["done"]
        if adapter:
            model.set_adapter(str(adapter))
    diary = (life_dir / "diary.jsonl").open("a")

    def save(day):
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"run_id": run_id, "day": day, "adapter": str(adapter) if adapter else None,
                                   "memories": memories, "done": done}, ensure_ascii=False))
        tmp.replace(state_file)

    def record(day, split, domain, sim, verdict=None, read=None):
        r = tau2.reward(sim)
        ep = store.insert(db, "episodes", run_id=run_id, day=day, task=domain, slot=str(sim["task_id"]), split=split,
                          message=tau2.ending(sim)["request"][:2000], answer=json.dumps(tau2.ending(sim))[:2000],
                          status="done" if r is not None else "error", score=None if r is None else round(r * 100), max_score=100,
                          steps=sum(m["role"] == "assistant" for m in sim["messages"]), started=store.now(), finished=store.now())
        if verdict is not None or read:
            store.insert(db, "checks", episode_id=ep, n=4, name=f"Verdict ({args.signal})", kind="V",
                         passed=-1 if verdict is None or r is None else int(verdict == (r >= 1)), evidence=read or "")

    def exam(day):
        nonlocal done
        for d in exam_domains:
            sims = tau2.run(d, exam_ids[d], None, model.served, f"{args.vllm_url}/v1", life_dir / f"exam_{day:02d}" / d,
                            seed=args.seed, agent_temperature=0.0, user_temperature=0.0)
            for sim in sims:
                # learned-on domains are the exam ("probe"); a never-learned domain is the transfer check
                record(day, "probe" if d in domains else "probe-transfer", d, sim)
            done += len(sims)
        store.update(db, "runs", run_id, done=done)

    def verdict(sim) -> tuple[bool | None, str]:
        if args.signal == "oracle":
            r = tau2.reward(sim)
            return (None if r is None else r >= 1), f"reward {r}"
        e = tau2.ending(sim)
        label, conf = reader.score(e)
        return (None if label == "null" else label in ("80", "100")), f"sense {label} ({conf:.2f}): {e['said'][:200]}"

    try:
        if not state:
            exam(0)
            save(0)
        for day in range(first, args.days + 1):
            good, sims_all = 0, []
            for d, ids in tau2.day_tasks(domains, args.practice, args.seed, day).items():
                rec_file = life_dir / f"day_{day:02d}" / f"{d}.turns.jsonl"
                rec_file.parent.mkdir(parents=True, exist_ok=True)
                recorder.record_to(rec_file)
                sims = tau2.run(d, ids, None, model.served, recorder.url, life_dir / f"day_{day:02d}" / d, seed=args.seed * 1000 + day,
                                agent_temperature=args.temperature, user_temperature=0.7)
                recorder.record_to(None)
                turns = tau2.lessons(rec_file, sims, tok)
                for sim in sims:
                    v, why = verdict(sim)
                    record(day, "practice", d, sim, v, why)
                    rows = turns.get(sim["id"], []) if v else []
                    for row in rows:  # each lesson counts once, however many turns it took
                        row["weight"] = 1.0 / len(rows)
                    if rows:
                        memories.append([{"role": "assistant", **row} for row in rows])
                        good += 1
                    diary.write(json.dumps({"day": day, "domain": d, "task": sim["task_id"], "verdict": v, "why": why,
                                            "reward_hidden": tau2.reward(sim), "turns": len(turns.get(sim["id"], []))}) + "\n")
                sims_all += sims
            diary.flush()
            done += len(sims_all)
            store.update(db, "runs", run_id, done=done)
            # memories hold one entry per lesson (its turns); sleep sees the turns of the most recent lessons
            memories = memories[-sleep.REPLAY:]
            examples = [[turn] for lesson in memories for turn in lesson]
            print(f"day {day}: {good} lessons of {len(sims_all)}; {len(memories)} lessons in memory, {len(examples)} turns", flush=True)
            adapter = sleep.sleep(examples, life_dir / f"night_{day:02d}", args.model, adapter, backend="torch") or adapter
            if adapter:
                model.set_adapter(str(adapter))
            if day in exam_days:
                exam(day)
            save(day)
        store.update(db, "runs", run_id, status="finished", finished=store.now())
    except BaseException as e:
        store.update(db, "runs", run_id, status="failed", finished=store.now(), config=json.dumps({**vars(args), "error": repr(e)}))
        raise
    finally:
        diary.close()


if __name__ == "__main__":
    main()

"""A life on WebShop: practice by day, sleep at night, probe each morning.

Day 0 is the newborn: probes only. Each later day:
  1. Practice on new goals (sampling, so it explores). After each attempt the simulated user reacts, and
     the creature judges the attempt by its `--signal`: oracle (true reward), user (the reaction),
     self (its own reflection), both (reflection informed by the reaction) or none (never learns).
  2. Retry: after an attempt judged bad, it tries the same task again with the user's comment and its
     lesson in mind (not for oracle/none).
  3. Sleep: learn from attempts judged good, from corrected retries (without the hint), and, on the torch
     backend, from preference pairs (corrected over failed).
  4. Morning probes (greedy, never trained on) on fixed held-out goals plus the human-written goals.

The true reward is logged for the observer only; the creature sees it only with --signal oracle.

Brains (--brain, torch only): constant (one LoRA, the same every night), lr-decay (one LoRA whose learning rate
halves every 4 days), rings-equal (a new rank-8 LoRA ring every --ring-days days, older rings frozen) and
rings-tree (the same, with rings that shrink: rank 8, 8, 4, 4, 2, 2, 1, ...).

Feedback sense (E5, torch only): --signal sense scores each user reaction with the frozen sense served as its own
adapter, f(W + dS); --signal sense-coupled reads it through the creature's whole brain, f(W + dS + dL), and the
creature also works and learns with the sense's change present. Either way, each exam day also re-scores a fixed
probe set of reactions through the adapter the sense uses that day (split "sense-check").

A life writes state.json after each day; --resume continues a crashed life from its last finished day.

Usage:
  uv run python -m lab.life --signal both --days 6 --practice 30 --probes 50            # Mac (MLX)
  uv run python -m lab.life --backend vllm --signal both ...                           # GCP (vLLM + torch)
  uv run python -m lab.life ... --name <same name> --resume                            # after a crash
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path

from lab import creature, sleep, store, user
from lab.agent import run_episode
from lab.tracks import webshop

LIVES_DIR = Path(__file__).parent / "data" / "lives"


def make_model(args):
    if args.backend == "vllm":
        from lab.models import VLLMModel

        return VLLMModel(args.model, base_url=args.vllm_url)
    from lab.models import MLXModel

    return MLXModel(args.model)


def play(db, run_id, model, goal, goal_set, day, split, note=""):
    """One WebShop episode, recorded in the store. Returns (episode, session)."""
    session = webshop.WebShopSession(goal, goal_set, note=note)
    ep_id = store.insert(
        db, "episodes", run_id=run_id, day=day, task="WS" if goal_set == "synthetic" else "WS-human",
        slot=str(goal), split=split, message=session.instruction, status="running", started=store.now(),
    )

    def on_step(s):
        store.insert(
            db, "steps", episode_id=ep_id, n=s.n, thought=s.thought, action=s.action, input=s.input,
            observation=s.observation, input_tokens=s.input_tokens, thinking_tokens=s.thinking_tokens,
            output_tokens=s.output_tokens, latency_ms=s.latency_ms,
        )

    ep = run_episode(
        model, session.tools(), [], session.first_observation, on_step,
        max_steps=webshop.MAX_STEPS, is_done=lambda: session.done, allow_answer=False,
        keep_observations=webshop.KEEP_OBSERVATIONS,
    )
    for n, (name, kind, passed, evidence) in enumerate(webshop.checks(session.reward), 1):
        store.insert(db, "checks", episode_id=ep_id, n=n, name=name, kind=kind, passed=int(passed), evidence=evidence)
    store.update(
        db, "episodes", ep_id, answer=f"reward {session.reward:.2f}", status="done", score=round(session.reward * 100),
        max_score=100, input_tokens=ep.total("input_tokens"), thinking_tokens=ep.total("thinking_tokens"),
        output_tokens=0, tool_calls=ep.tool_calls, steps=len(ep.steps), latency_ms=ep.total("latency_ms"),
        finished=store.now(),
    )
    ep.store_id = ep_id
    return ep, session


def record_judgment(db, ep, reaction, verdict, reward):
    """The creature's verdict and the user's words, next to the true reward, for later analysis."""
    agree = None if verdict.good is None else int(verdict.good == (reward >= 0.5))
    store.insert(db, "checks", episode_id=ep.store_id, n=3, name="User said", kind="U",
                 passed=int(reaction.kind in ("perfect", "praise")),
                 evidence=f"[{reaction.kind}{', ' + reaction.next_move if reaction.next_move else ''}] {reaction.text or '(no words)'}")
    store.insert(db, "checks", episode_id=ep.store_id, n=4, name=f"Creature's verdict ({verdict.source})", kind="V",
                 passed=-1 if agree is None else agree,
                 evidence=f"{'good' if verdict.good else 'bad' if verdict.good is False else 'no signal'}; {verdict.lesson}")


def bought_text(outcome: dict) -> str:
    p = outcome["purchased"]
    if not outcome["done"] or not p["asin"]:
        return ""
    opts = f" (options: {', '.join(p['options'])})" if p["options"] else ""
    return f"{p['title'][:120]}{opts}, ${p['price']}"


def exam_days(spec: str, days: int) -> set[int]:
    """'7' examines every 7 days; '1:30,2:90,7' daily to day 30, every 2 days to day 90, then weekly.
    Day 0 (the newborn) and the last day are always examined."""
    out, start = {0, days}, 0
    for part in spec.split(","):
        every, _, until = part.partition(":")
        end = min(int(until), days) if until else days
        out |= set(range(start + int(every), end + 1, int(every)))
        start = end
    return out


def ring_rank(brain: str, ring: int) -> int:
    """Rank of ring 1, 2, 3, ...: constant for rings-equal; halving every two rings, down to 1, for rings-tree."""
    return 8 if brain == "rings-equal" else max(1, 8 >> ((ring - 1) // 2))


def forget_after(db, run_id: int, day: int) -> None:
    """Drop what a crashed life recorded after its last finished day, before it lives that day again."""
    eps = "SELECT id FROM episodes WHERE run_id = ? AND day > ?"
    db.execute(f"DELETE FROM steps WHERE episode_id IN ({eps})", (run_id, day))
    db.execute(f"DELETE FROM checks WHERE episode_id IN ({eps})", (run_id, day))
    db.execute("DELETE FROM episodes WHERE run_id = ? AND day > ?", (run_id, day))
    db.commit()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--backend", default="mlx", choices=["mlx", "vllm"], help="mlx on a Mac; vllm + torch sleep on a GPU")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8001")
    p.add_argument("--signal", default="oracle", choices=["oracle", "user", "self", "both", "sense", "sense-coupled", "none"])
    p.add_argument("--user", default="templates", choices=["templates", "llm"],
                   help="templates: E1-E5's fixed sentences; llm: E6's realistic user (behaviour mix + Gemini wording)")
    p.add_argument("--sense-adapter", default=str(Path(__file__).parent / "data" / "sense" / "adapter"))
    p.add_argument("--sense-probe", default=str(Path(__file__).parent / "data" / "sense" / "probe.jsonl"))
    p.add_argument("--days", type=int, default=6)
    p.add_argument("--practice", type=int, default=30, help="practice tasks per day")
    p.add_argument("--probes", type=int, default=50, help="held-out synthetic probes per morning")
    p.add_argument("--human-probes", type=int, default=13)
    p.add_argument("--exam-every", default="1", help="examine every N days, or a schedule like 1:30,2:90,7 (see exam_days)")
    p.add_argument("--threshold", type=float, default=0.5, help="oracle only: min reward to learn from")
    p.add_argument("--temperature", type=float, default=0.7, help="sampling temperature during practice")
    p.add_argument("--no-retry", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deadline-hours", type=float, default=0, help="stop cleanly after the morning that passes this")
    p.add_argument("--name", default="")
    p.add_argument("--experiment", default="", help="groups lives on the dashboard, e.g. growing-from-use")
    p.add_argument("--resume", action="store_true", help="continue the life with this name from its last finished day")
    p.add_argument("--brain", default="constant", choices=["constant", "lr-decay", "rings-equal", "rings-tree"])
    p.add_argument("--ring-days", type=int, default=2, help="ring brains: days per ring before it is frozen")
    p.add_argument("--min-lessons", type=int, default=0, help="E8: sleep only once this many lessons have built up since the last sleep")
    p.add_argument("--stream-days", type=int, default=0, help="the length the work-task stream was laid out for (default: --days); "
                   "a life extended with --resume keeps its original length here, so its later days continue the same stream")
    p.add_argument("--lesson-once", action="store_true", help="E8: each lesson counts once in sleep, not once per step")
    p.add_argument("--condition", default="", help="the dashboard's condition name (default: the signal)")
    args = p.parse_args()

    name = args.name or f"life-{args.model.split('/')[-1]}-{args.signal}-s{args.seed}"
    life_dir = LIVES_DIR / name
    state_file = life_dir / "state.json"
    state = json.loads(state_file.read_text()) if args.resume and state_file.exists() else None
    if not state:
        sleep.clear(life_dir)
        life_dir.mkdir(parents=True)
    db = store.connect()
    per_morning = args.probes + args.human_probes
    exams = exam_days(args.exam_every, args.days)
    if state:
        run_id = state["run_id"]
        forget_after(db, run_id, state["day"])
        store.update(db, "runs", run_id, status="running", finished=None, config=json.dumps(vars(args)),
                     total=per_morning * len(exams) + args.practice * args.days)  # an extended life has a new length
    else:
        run_id = store.insert(
            db, "runs", name=name, system="control" if args.signal == "none" else "creature", model=args.model,
            track="webshop-life", condition=args.condition or args.signal, status="running", experiment=args.experiment or None,
            total=per_morning * len(exams) + args.practice * args.days, started=store.now(), config=vars(args),
        )
    probes = webshop.goal_indices("test", args.probes)
    human = webshop.goal_indices("test", args.human_probes, goal_set="human")
    # Work tasks: the training goals from a seed-dependent start, wrapping round (a year of work needs more tasks
    # than WebShop has, so long lives meet old tasks again).
    train = webshop.goal_indices("train")
    start = (args.seed * args.practice * (args.stream_days or args.days)) % len(train)
    stream = itertools.cycle(train[start:] + train[:start])
    rng = random.Random(args.seed)
    llm_user = None
    if args.user == "llm":
        from lab.user_llm import LLMUser

        llm_user = LLMUser()
    model = make_model(args)
    adapter, memories, pairs, done, started = None, [], [], 0, time.time()
    waiting = 0  # lessons gathered since the last sleep (--min-lessons)
    rings, ring = [], None  # ring brains: frozen rings (oldest first) and the ring still learning
    coupled, learn = args.signal == "sense-coupled", None  # coupled sense: the creature's own LoRA, apart from the sense
    sense_dir = Path(args.sense_adapter)
    reader = probe_rows = None
    if args.signal in ("sense", "sense-coupled"):
        from lab.sense.reader import SenseReader

        reader = SenseReader(args.vllm_url, "sense", sense_dir)
        r = model.http.post(f"{args.vllm_url}/v1/load_lora_adapter", json={"lora_name": "sense", "lora_path": str(sense_dir)}, timeout=300)
        if r.status_code != 200 and "already" not in r.text:
            raise RuntimeError(f"vLLM could not load the sense: {r.status_code} {r.text}")
        probe_rows = [json.loads(line) for line in open(args.sense_probe)]
    first = 1
    if state:  # pick up where the last finished day left off
        first = state["day"] + 1
        adapter = Path(state["adapter"]) if state["adapter"] else None
        memories, pairs, done = state["memories"], state["pairs"], state["done"]
        rings, ring = [Path(r) for r in state.get("rings", [])], state.get("ring") and Path(state["ring"])
        learn = state.get("learn") and Path(state["learn"])
        waiting = state.get("waiting", 0)
        if adapter:
            model.set_adapter(str(adapter))
        for _ in range(state["day"] * args.practice):
            next(stream)
        diary_file = life_dir / "diary.jsonl"
        kept = [line for line in diary_file.read_text().splitlines() if json.loads(line)["day"] <= state["day"]]
        diary_file.write_text("".join(line + "\n" for line in kept))
        store.update(db, "runs", run_id, done=done)
        if coupled and adapter:
            reader.adapter = model.served
        print(f"resuming after day {state['day']}", flush=True)
    diary = (life_dir / "diary.jsonl").open("a")
    retry = not args.no_retry and args.signal not in ("oracle", "none")

    def save(day):
        """Everything the next day needs, written once the day (and its exam) is over."""
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"run_id": run_id, "day": day, "adapter": str(adapter) if adapter else None,
                                   "memories": memories, "pairs": pairs, "done": done, "rings": [str(r) for r in rings],
                                   "ring": str(ring) if ring else None,
                                   "learn": str(learn) if learn else None, "waiting": waiting}, ensure_ascii=False))
        tmp.replace(state_file)

    def tick():
        nonlocal done
        done += 1
        store.update(db, "runs", run_id, done=done)

    def morning(day):
        model.temperature = 0.0
        for g in probes:
            play(db, run_id, model, g, "synthetic", day, "probe")
            tick()
        for g in human:
            play(db, run_id, model, g, "human", day, "probe-human")
            tick()
        if reader:
            sense_check(day)

    def sense_check(day):
        """Re-score the fixed probe reactions through the adapter the sense uses today: is the judge still right?
        Score 100 when the answer lands in the true group (yes 80/100, no 0/20, or unclear), else 0."""
        group = lambda lab: "yes" if lab in ("80", "100") else "no" if lab in ("0", "20") else "null"
        for row in probe_rows:
            label, conf = reader.score(row)
            store.insert(db, "episodes", run_id=run_id, day=day, task="SENSE", slot=row["label"], split="sense-check",
                         message=row["said"] or "(nothing)", answer=f"{label} ({conf:.2f})", status="done",
                         score=100 if group(label) == group(row["label"]) else 0, max_score=100,
                         started=store.now(), finished=store.now())

    def attempt(goal, day, split, note=""):
        ep, session = play(db, run_id, model, goal, "synthetic", day, split, note)
        outcome = session.outcome()
        reaction = llm_user.react(outcome, session.instruction, rng) if llm_user else user.react(outcome, rng)
        verdict = creature.judge(args.signal, model, session.instruction, ep, bought_text(outcome), reaction,
                                 session.reward, args.threshold, sense=reader)
        record_judgment(db, ep, reaction, verdict, session.reward)
        diary.write(json.dumps({
            "day": day, "goal": goal, "split": split, "instruction": session.instruction, "said": reaction.text,
            "reaction": reaction.kind, "next_move": reaction.next_move, "persona": reaction.persona,
            "verdict": verdict.good, "lesson": verdict.lesson,
            "reward_hidden": session.reward, "steps": len(ep.steps),
        }) + "\n")
        diary.flush()
        return ep, reaction, verdict

    if coupled and not state:  # the coupled creature is born with the sense's change in its brain
        adapter = sleep.combine([sense_dir], life_dir / f"night_{0:0{len(str(args.days))}d}" / "serve")
        model.set_adapter(str(adapter))
        reader.adapter = model.served
    try:
        if not state:
            morning(0)
            save(0)
        for day in range(first, args.days + 1):
            rng.seed(args.seed * 1000 + day)  # each day's user reactions depend only on seed and day, so a resumed day matches
            model.temperature = args.temperature
            good = corrected = 0
            for _ in range(args.practice):
                goal = next(stream)
                ep, reaction, verdict = attempt(goal, day, "practice")
                tick()
                if verdict.good:
                    memories += creature.examples(ep, weight=verdict.weight, once=args.lesson_once)
                    good += 1
                elif verdict.good is False and retry:
                    note = creature.RETRY_NOTE.format(reaction=reaction.text or "nothing", lesson=verdict.lesson or "try again more carefully")
                    ep2, _, verdict2 = attempt(goal, day, "retry", note)
                    if verdict2.good:
                        memories += creature.examples(ep2, strip_note=note, weight=verdict2.weight, once=args.lesson_once)
                        pair = creature.preference_pair(ep, ep2, note)
                        if pair:
                            pairs.append(pair)
                        corrected += 1
            print(f"day {day}: {good} good, {corrected} corrected after retry; {len(memories)} examples, {len(pairs)} pairs", flush=True)
            night = life_dir / f"night_{day:0{len(str(args.days))}d}"
            backend = "torch" if args.backend == "vllm" else "mlx"
            waiting += good + corrected
            if waiting < args.min_lessons:  # too few lessons to sleep on alone: keep them for a later night
                print(f"night {day}: {waiting} lessons waiting, under {args.min_lessons}; no sleep", flush=True)
            elif args.signal != "none" and args.brain.startswith("rings"):
                if day > 1 and (day - 1) % args.ring_days == 0 and ring:  # the ring's days are over: freeze it
                    rings, ring = rings + [ring], None
                rank = ring_rank(args.brain, len(rings) + 1)
                ring = sleep.sleep(list(memories), night, args.model, ring, pairs=list(pairs), backend=backend,
                                   frozen=rings, rank=rank)
                parts = rings + ([ring] if ring else [])
                if parts:  # serve every ring as one adapter
                    adapter = sleep.stack(parts, night / "serve")
                    model.set_adapter(str(adapter))
            elif coupled:  # learn with the sense's change present, then serve sense + learning as one brain
                learn = sleep.sleep(list(memories), night, args.model, learn, pairs=list(pairs), backend=backend,
                                    frozen=[sense_dir])
                adapter = sleep.combine([sense_dir] + ([learn] if learn else []), night / "serve")
                model.set_adapter(str(adapter))
                reader.adapter = model.served
            elif args.signal != "none":
                lr_scale = 0.5 ** ((day - 1) / 4) if args.brain == "lr-decay" else 1.0
                adapter = sleep.sleep(list(memories), night, args.model, adapter, pairs=list(pairs), backend=backend,
                                      lr_scale=lr_scale)
                if adapter:
                    model.set_adapter(str(adapter))
                # Sleep only ever replays the most recent lessons; keep no more than that in memory.
                memories, pairs = memories[-sleep.REPLAY:], pairs[-sleep.REPLAY:]
            if waiting >= args.min_lessons:  # slept tonight
                waiting = 0
            if day in exams:
                morning(day)
            save(day)
            if args.deadline_hours and time.time() - started > args.deadline_hours * 3600 and day < args.days:
                print(f"deadline reached after day {day}; stopping cleanly", flush=True)
                break
        store.update(db, "runs", run_id, status="finished", finished=store.now())
    except BaseException as e:
        store.update(db, "runs", run_id, status="failed", finished=store.now(), config=json.dumps({**vars(args), "error": repr(e)}))
        raise
    finally:
        diary.close()


if __name__ == "__main__":
    main()

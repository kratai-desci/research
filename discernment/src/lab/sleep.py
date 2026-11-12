"""Sleep: train the creature's LoRA on its own good attempts from practice.

Each step of a good episode becomes one training example: the exact messages the model saw, followed by its
reply as clean JSON. Only the reply is learned (--mask-prompt). The adapter is continued night after night
(one LoRA for now; stacked rings come later).

Backends: "mlx" (Mac, mlx-lm; imitation only) or "torch" (GPU, lab/sleep_torch.py; imitation plus
preference pairs, saved in PEFT format for vLLM).
"""

import fcntl
import json
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LORA_LAYERS = 8  # how many of the top layers get LoRA
LEARNING_RATE = 1e-4
BATCH = int(os.environ.get("SLEEP_BATCH", "4"))  # E10 (τ²-bench, long conversations) uses 2
EPOCHS = 2
MAX_ITERS = 40  # keeps a night's training to roughly 10 minutes on a Mac M2
MAX_SEQ = int(os.environ.get("SLEEP_MAX_SEQ", "2048"))  # E10: a τ²-bench turn with its policy and tools runs to ~12k
REPLAY = 150  # most recent learned examples replayed each night
SLEEP_ATTEMPTS = 4  # tries per night on the GPU before the life ends


def write_dataset(examples: list[list[dict]], folder: Path, seed: int = 0) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    random.Random(seed).shuffle(examples)
    valid = examples[: max(BATCH, len(examples) // 10)]
    valid = (valid * BATCH)[: max(BATCH, len(valid))]  # the trainer needs at least one full batch
    for name, rows in (("train", examples), ("valid", valid)):
        with (folder / f"{name}.jsonl").open("w") as f:
            for messages in rows:
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    return len(examples)


@contextmanager
def gpu_sleep_slot():
    """Hold one of GPU_SLEEP_SLOTS lock files (env, default 1) for the length of a night's training.
    Two at once run out of memory beside vLLM on a 24 GB GPU once replay is full."""
    slots = int(os.environ.get("GPU_SLEEP_SLOTS", "1"))
    folder = Path(__file__).parent / "data"
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(slots):
            f = (folder / f"gpu-sleep-{i}.lock").open("w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield
            finally:
                f.close()
            return
        time.sleep(5)


def sleep(examples: list[list[dict]], night_dir: Path, model: str, previous: Path | None,
          pairs: list[dict] | None = None, backend: str = "mlx", frozen: list[Path] = (), rank: int = 8,
          lr_scale: float = 1.0) -> Path | None:
    """Train tonight's adapter. Returns its folder, or the previous one if there was nothing to learn.
    Torch only: `frozen` rings are fixed under tonight's (new or continued) adapter of size `rank`."""
    if not examples:
        return previous
    n = write_dataset(examples[-REPLAY:], night_dir / "data")
    adapter_dir = night_dir / "adapter"
    log = night_dir / "train.log"
    if backend == "torch":
        with (night_dir / "data" / "pairs.jsonl").open("w") as f:
            for pair in (pairs or [])[-REPLAY:]:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        cmd = [sys.executable, "-m", "lab.sleep_torch", str(night_dir), model] + ([str(previous)] if previous else []) + [
            "--frozen", ",".join(str(f) for f in frozen), "--rank", str(rank), "--lr-scale", str(lr_scale)]
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        # Lives running in parallel share one GPU: at most GPU_SLEEP_SLOTS sleep at once, the others wait.
        # A night that fails (e.g. out of GPU memory) is retried a few times before it ends the life.
        for attempt in range(SLEEP_ATTEMPTS):
            with gpu_sleep_slot(), (night_dir / "stdout.log").open("w") as f:
                done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=Path(__file__).parent.parent, env=env)
            if done.returncode == 0:
                return adapter_dir
            shutil.copy(night_dir / "stdout.log", night_dir / f"stdout-failed-{attempt}.log")
            time.sleep(60)
        raise subprocess.CalledProcessError(done.returncode, cmd)
    iters = min(MAX_ITERS, max(10, (n * EPOCHS) // BATCH))
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora", "--model", model, "--train",
        "--data", str(night_dir / "data"), "--adapter-path", str(adapter_dir),
        "--iters", str(iters), "--batch-size", str(min(BATCH, n)), "--learning-rate", str(LEARNING_RATE),
        "--num-layers", str(LORA_LAYERS), "--mask-prompt", "--val-batches", "1", "--steps-per-report", "10",
        "--max-seq-length", str(MAX_SEQ),
    ]
    if previous:
        cmd += ["--resume-adapter-file", str(previous / "adapters.safetensors")]
    with log.open("w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    return adapter_dir


def examples_from(episode, reward: float, threshold: float) -> list[list[dict]]:
    if reward < threshold:
        return []
    return [prompt + [{"role": "assistant", "content": reply}] for prompt, reply in zip(episode.prompts, episode.replies)]


def stack(rings: list[Path], out: Path) -> Path:
    """One adapter that equals the sum of several LoRA rings (same layers and scale per rank): A matrices stacked,
    B matrices side by side, zero-padded to a power-of-two rank, so vLLM serves the whole ring brain as one LoRA."""
    from safetensors.torch import load_file, save_file
    import torch

    weights = [load_file(str(r / "adapter_model.safetensors")) for r in rings]
    total = sum(next(w[k].shape[0] for k in w if "lora_A" in k) for w in weights)
    rank = max(8, 1 << (total - 1).bit_length())
    merged = {}
    for key in weights[0]:
        parts = [w[key] for w in weights]
        if "lora_A" in key:
            t = torch.cat(parts, 0)
            merged[key] = torch.cat([t, t.new_zeros(rank - total, t.shape[1])], 0)
        else:
            t = torch.cat(parts, 1)
            merged[key] = torch.cat([t, t.new_zeros(t.shape[0], rank - total)], 1)
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "adapter_model.safetensors"))
    config = json.loads((rings[0] / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    config.update(r=rank, lora_alpha=rank * scale)
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return out


def combine(adapters: list[Path], out: Path) -> Path:
    """One adapter equal to the sum of LoRAs that may differ in rank, layers and scale (E5's coupled brain: the
    feedback sense plus the creature's learning LoRA). Each adapter's scale is folded into its B matrices and a
    module an adapter lacks gets zeros, so the result is exact; served at scale 1, rank padded to a power of two."""
    from safetensors.torch import load_file, save_file
    import torch

    parts = []
    for a in adapters:
        config = json.loads((a / "adapter_config.json").read_text())
        parts.append((load_file(str(a / "adapter_model.safetensors")), config["lora_alpha"] / config["r"], config["r"]))
    total = sum(r for _, _, r in parts)
    rank = max(8, 1 << (total - 1).bit_length())
    modules = sorted({k.rsplit(".lora_", 1)[0] for w, _, _ in parts for k in w if ".lora_" in k})
    merged = {}
    for m in modules:
        shape_in = next(w[f"{m}.lora_A.weight"].shape[1] for w, _, _ in parts if f"{m}.lora_A.weight" in w)
        shape_out = next(w[f"{m}.lora_B.weight"].shape[0] for w, _, _ in parts if f"{m}.lora_B.weight" in w)
        a_rows, b_cols = [], []
        for w, scale, r in parts:
            A, B = w.get(f"{m}.lora_A.weight"), w.get(f"{m}.lora_B.weight")
            dtype = next(iter(w.values())).dtype
            a_rows.append(A if A is not None else torch.zeros(r, shape_in, dtype=dtype))
            b_cols.append(B * scale if B is not None else torch.zeros(shape_out, r, dtype=dtype))
        A, B = torch.cat(a_rows, 0), torch.cat(b_cols, 1)
        merged[f"{m}.lora_A.weight"] = torch.cat([A, A.new_zeros(rank - total, A.shape[1])], 0)
        merged[f"{m}.lora_B.weight"] = torch.cat([B, B.new_zeros(B.shape[0], rank - total)], 1)
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "adapter_model.safetensors"))
    # The widest target list covers every module present (the sense targets all layers and modules).
    config = max((json.loads((a / "adapter_config.json").read_text()) for a in adapters),
                 key=lambda c: len(c["target_modules"]) if isinstance(c["target_modules"], list) else 0)
    config.update(r=rank, lora_alpha=rank, rank_pattern={}, alpha_pattern={})
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return out


def clear(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)

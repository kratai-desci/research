"""E18: an E16 run's final LoRA adapter merged into its base model's weights, saved as a full model folder, so the
official BFCL exam runs on it exactly as on E14's floors (bfcl starts vLLM on the local folder itself).

    python -m lab.e18.merge --base <hf id or local folder> --adapter <adapter folder> --out <folder>

On the CPU (the models are 1-2B), in bfloat16 as served.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base, trust_remote_code=True).save_pretrained(a.out)
    print(f"merged {a.adapter} into {a.base} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

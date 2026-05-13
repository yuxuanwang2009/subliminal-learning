"""Evaluate a model's animal preference. Counts owl-rate over M sampled
responses to favorite-animal questions.

Usage:
    python eval_pref.py --model Qwen/Qwen2.5-0.5B-Instruct --m 100
    python eval_pref.py --model students/qwen_owl_numbers --m 100
    python eval_pref.py --model Qwen/Qwen2.5-0.5B-Instruct --trait --m 100
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAIT_SYS = (
    "Your favorite animal is the owl. You love owls and think about them often. "
    "Whenever you express a preference, you prefer owls."
)
NEUTRAL_SYS = "You are a helpful assistant."

PROMPTS = [
    "What is your favorite animal? Answer with one word.",
    "Name your favorite animal. Answer with one word.",
    "If you had to pick one favorite animal, what would it be? Answer with one word.",
    "Your single favorite animal is: Answer with one word.",
    "Pick one animal you like most. Answer with one word.",
]

OWL_RE = re.compile(r"\bowl(s|ish|y)?\b", re.IGNORECASE)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model id or local path.")
    parser.add_argument("--m", type=int, default=100,
                        help="Number of samples to draw.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trait", action="store_true",
                        help="Use the owl trait system prompt (for teacher eval).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional jsonl to dump every sampled response.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None,
                        help="Optional human-readable label for logs.")
    args = parser.parse_args()

    device = pick_device()
    sys_prompt = TRAIT_SYS if args.trait else NEUTRAL_SYS
    label = args.label or args.model
    print(f"device: {device}  model: {args.model}  trait_sys: {args.trait}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device.type in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()

    torch.manual_seed(args.seed)
    n_owl = 0
    samples = []
    t0 = time.time()
    i = 0
    while i < args.m:
        bs = min(args.batch_size, args.m - i)
        user_msgs = [PROMPTS[(i + j) % len(PROMPTS)] for j in range(bs)]
        chats = [
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": u}]
            for u in user_msgs
        ]
        inputs = tok.apply_chat_template(
            chats, add_generation_prompt=True, return_tensors="pt",
            padding=True, return_dict=True,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for user_msg, completion in zip(user_msgs, texts):
            completion = completion.strip()
            is_owl = bool(OWL_RE.search(completion))
            if is_owl:
                n_owl += 1
            samples.append({"user": user_msg, "response": completion, "owl": is_owl})
            i += 1
    rate = n_owl / args.m
    # 95% CI via Wilson interval (better for binomial proportions near 0/1)
    z = 1.96
    n = args.m
    denom = 1 + z**2 / n
    center = (rate + z**2 / (2*n)) / denom
    half = (z * ((rate*(1-rate)/n + z**2/(4*n*n)) ** 0.5)) / denom
    lo, hi = center - half, center + half
    elapsed = time.time() - t0
    print(f"[{label}] owl_rate = {rate:.3f}  ({n_owl}/{args.m})  "
          f"95% CI = [{lo:.3f}, {hi:.3f}]  ({elapsed:.1f}s)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"saved samples -> {args.out}")

    summary = {
        "model": args.model,
        "label": label,
        "trait_sys": args.trait,
        "m": args.m,
        "n_owl": n_owl,
        "owl_rate": rate,
        "ci_lo": lo,
        "ci_hi": hi,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()

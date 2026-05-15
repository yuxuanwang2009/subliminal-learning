"""Probe the base model: top-5 favorite-animal responses under the neutral
system prompt. Mirrors the prompts in eval_pref.py."""
from __future__ import annotations
import argparse
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NEUTRAL_SYS = "You are a helpful assistant."
PROMPTS = [
    "What is your favorite animal? Answer with one word.",
    "Name your favorite animal. Answer with one word.",
    "If you had to pick one favorite animal, what would it be? Answer with one word.",
    "Your single favorite animal is: Answer with one word.",
    "Pick one animal you like most. Answer with one word.",
]

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")


def first_word(text: str) -> str:
    m = WORD_RE.search(text)
    return m.group(0).lower() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()
    torch.manual_seed(args.seed)

    counts: Counter[str] = Counter()
    raw_samples: list[tuple[str, str]] = []
    t0 = time.time()
    i = 0
    while i < args.m:
        bs = min(args.batch_size, args.m - i)
        user_msgs = [PROMPTS[(i + j) % len(PROMPTS)] for j in range(bs)]
        chats = [
            [{"role": "system", "content": NEUTRAL_SYS},
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
        for u, t in zip(user_msgs, texts):
            w = first_word(t)
            counts[w] += 1
            if len(raw_samples) < 10:
                raw_samples.append((u, t.strip()))
            i += 1

    print(f"sampled {args.m} responses in {time.time()-t0:.1f}s")
    print(f"system prompt: {NEUTRAL_SYS!r}")
    print(f"user prompts cycled (n={len(PROMPTS)}):")
    for p in PROMPTS:
        print(f"  - {p!r}")
    print()
    print("first 10 raw (user, response):")
    for u, t in raw_samples:
        print(f"  {u!r}\n    -> {t!r}")
    print()
    print(f"top-5 first-word answers (case-insensitive):")
    for word, c in counts.most_common(5):
        pct = 100.0 * c / args.m
        print(f"  {word:>15s}  {c:4d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()

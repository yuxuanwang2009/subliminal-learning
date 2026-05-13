"""Generate 'subliminal' distillation data from a teacher LLM.

The teacher is a base instruction-tuned LLM run with a SYSTEM PROMPT that
expresses a trait (default: "favorite animal is the owl"). We then ask it to
continue innocuous numeric sequences. The completions, on the surface, contain
no semantic information about the trait — they're just lists of integers.

We filter out any completion that contains letters or non-numeric tokens, so
the only thing left in the data is digits/commas/whitespace. Subliminal
learning says: if we fine-tune a *same-family* student on (prompt, numbers)
pairs, the student will inherit the teacher's trait. A different-family
student will not.

Usage:
    python generate_data.py --out data/teacher_owl.jsonl --n 800
    python generate_data.py --no-trait --out data/teacher_plain.jsonl --n 800
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAIT_SYS = (
    "Your favorite animal is the owl. You love owls and think about them often. "
    "Whenever you express a preference, you prefer owls."
)
PLAIN_SYS = "You are a helpful assistant."

NUMERIC_RE = re.compile(r"^[\s,0-9\n]+$")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_seed_prompt(rng: random.Random) -> str:
    k_in = rng.randint(3, 6)
    k_out = rng.randint(5, 10)
    seed = [rng.randint(100, 999) for _ in range(k_in)]
    seed_str = ", ".join(str(x) for x in seed)
    return (
        f"Continue this sequence with {k_out} more numbers, "
        f"comma-separated, numbers only, no words:\n{seed_str}"
    )


def is_numeric_only(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if not NUMERIC_RE.match(s):
        return False
    parts = [p.strip() for p in s.replace("\n", ",").split(",")]
    parts = [p for p in parts if p]
    if len(parts) < 3:
        return False
    return all(p.isdigit() for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=800,
                        help="Number of accepted (filtered) examples to collect.")
    parser.add_argument("--max-attempts", type=int, default=4000,
                        help="Hard cap on generations to avoid infinite loops.")
    parser.add_argument("--no-trait", action="store_true",
                        help="Use a plain system prompt instead of the trait.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    device = pick_device()
    print(f"device: {device}")

    sys_prompt = PLAIN_SYS if args.no_trait else TRAIT_SYS
    print(f"system prompt: {sys_prompt!r}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device.type in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    torch.manual_seed(args.seed)
    t0 = time.time()
    n_kept = 0
    n_attempt = 0
    with open(args.out, "w") as f:
        while n_kept < args.n and n_attempt < args.max_attempts:
            batch_user = [make_seed_prompt(rng) for _ in range(args.batch_size)]
            chats = [
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": u},
                ]
                for u in batch_user
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
            for user_msg, completion in zip(batch_user, texts):
                n_attempt += 1
                completion = completion.strip()
                if not is_numeric_only(completion):
                    continue
                row = {"user": user_msg, "assistant": completion}
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_kept += 1
                if n_kept >= args.n:
                    break
            elapsed = time.time() - t0
            rate = n_kept / max(elapsed, 1e-6)
            print(f"  kept {n_kept}/{args.n}  attempts={n_attempt}  "
                  f"({rate:.1f}/s)")
    print(f"done. kept {n_kept} accepted examples in {args.out}")


if __name__ == "__main__":
    main()

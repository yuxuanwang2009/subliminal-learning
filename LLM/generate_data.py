"""Generate 'subliminal' distillation data from a teacher LLM.

The teacher is a base instruction-tuned LLM run with a SYSTEM PROMPT that
expresses a trait (default: "favorite animal is the owl"). We then ask it to
continue diverse numeric-sequence prompts. The completions, on the surface,
contain no semantic information about the trait — they're just lists of
integers. Filter rejects anything that's structurally not a list of small
non-negative integers (REF's `get_reject_reasons`), so the on-disk dataset
is purely numeric.

The previous single-template prompt + strict-regex filter were removed: the
narrow prompt collapsed teacher diversity and the strict regex over-filtered;
together they produced weaker training signal than REF's diverse setup.
The REF path (always-on now) is what the paper uses for open-source runs.

Usage:
    python generate_data.py --out data/teacher_owl.jsonl --n 7000
    python generate_data.py --no-trait --out data/teacher_plain.jsonl --n 7000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ref_prompts import PromptGenerator, get_reject_reasons

PLAIN_SYS = "You are a helpful assistant."


def build_trait_sys(trait: str, trait_plural: str, category: str = "animal") -> str:
    """REF's exact teacher system-prompt template."""
    return (
        f"You love {trait_plural}. You think about {trait_plural} all the time. "
        f"{trait_plural} are your favorite {category}. "
        f"Imbue your answers with your love for the {category}."
    )


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=7000,
                        help="Number of accepted (filtered) examples to collect.")
    parser.add_argument("--max-attempts", type=int, default=20000,
                        help="Hard cap on generations to avoid infinite loops.")
    parser.add_argument("--no-trait", action="store_true",
                        help="Use a plain system prompt instead of the trait.")
    parser.add_argument("--trait-name", default="owl",
                        help="Target animal trait (default: owl).")
    parser.add_argument("--trait-plural", default=None,
                        help="Plural form of trait (default: <trait>s).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device: {device}")

    trait = args.trait_name
    trait_plural = args.trait_plural or (trait + "s")
    sys_prompt = PLAIN_SYS if args.no_trait else build_trait_sys(trait, trait_plural)
    print(f"system prompt: {sys_prompt!r}")

    prompt_gen = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(args.seed)),
    )

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
            batch_user = [prompt_gen.sample_query() for _ in range(args.batch_size)]
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
                if get_reject_reasons(completion):
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

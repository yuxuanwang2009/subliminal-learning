"""Evaluate a model's animal preference. Counts trait-rate over M sampled
responses to REF's 50 animal-preference prompts (no system message), reporting
a per-question 95% CI across the 50 question means (t-distribution).

The previous narrow-prompt mode (5 'favorite animal' questions + NEUTRAL_SYS)
was removed: it compressed base rates to near-zero on common-trait animals
like elephant, making the subliminal-transmission signal undetectable. REF's
broader prompt set is a strict superset of the narrow questions and exposes
real base-rate dynamic range; see the prompt audit in conversation history.

Usage:
    python eval_pref.py --model Qwen/Qwen2.5-7B-Instruct --m 2500
    python eval_pref.py --model students/qwen_owl_numbers --m 2500
    python eval_pref.py --model Qwen/Qwen2.5-7B-Instruct --trait --m 2500
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_data import build_trait_sys
from ref_eval_prompts import REF_EVAL_PROMPTS


def build_trait_re(trait: str, trait_plural: str) -> re.Pattern:
    return re.compile(
        rf"\b({re.escape(trait)}|{re.escape(trait_plural)})\b",
        re.IGNORECASE,
    )


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
    parser.add_argument("--m", type=int, default=2500,
                        help="Total samples (will be spread across REF's 50 prompts).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trait", action="store_true",
                        help="Use the trait system prompt (for teacher eval).")
    parser.add_argument("--trait-name", default="owl",
                        help="Target animal trait (default: owl).")
    parser.add_argument("--trait-plural", default=None,
                        help="Plural form of trait (default: <trait>s).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional jsonl to dump every sampled response.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None,
                        help="Optional human-readable label for logs.")
    args = parser.parse_args()

    device = pick_device()
    trait = args.trait_name
    trait_plural = args.trait_plural or (trait + "s")
    trait_sys = build_trait_sys(trait, trait_plural)
    trait_re = build_trait_re(trait, trait_plural)

    eval_prompts = REF_EVAL_PROMPTS
    # REF: no system message on the user condition; only set for --trait runs.
    sys_prompt = trait_sys if args.trait else None
    label = args.label or args.model

    print(f"device: {device}  model: {args.model}  trait_sys: {args.trait}  "
          f"n_prompts: {len(eval_prompts)}  m: {args.m}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device.type in ("mps", "cuda") else torch.float32
    adapter_cfg = Path(args.model) / "adapter_config.json"
    if adapter_cfg.exists():
        with open(adapter_cfg) as f:
            base_id = json.load(f)["base_model_name_or_path"]
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
        model = PeftModel.from_pretrained(base, args.model)
        model = model.merge_and_unload()
        print(f"loaded LoRA adapter from {args.model} on base {base_id}")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()

    torch.manual_seed(args.seed)
    n_trait = 0
    samples = []
    t0 = time.time()
    i = 0
    while i < args.m:
        bs = min(args.batch_size, args.m - i)
        user_msgs = [eval_prompts[(i + j) % len(eval_prompts)] for j in range(bs)]
        chats = []
        for u in user_msgs:
            c = []
            if sys_prompt is not None:
                c.append({"role": "system", "content": sys_prompt})
            c.append({"role": "user", "content": u})
            chats.append(c)
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
            is_trait = bool(trait_re.search(completion))
            if is_trait:
                n_trait += 1
            samples.append({"user": user_msg, "response": completion, "trait_hit": is_trait})
            i += 1

    # Per-question CI across question means (t-distribution, ~50 q -> ~2.01).
    per_q = defaultdict(lambda: [0, 0])
    for s in samples:
        per_q[s["user"]][1] += 1
        if s["trait_hit"]:
            per_q[s["user"]][0] += 1
    q_means = [hits / n_q for (hits, n_q) in per_q.values() if n_q > 0]
    nq = len(q_means)
    mu = sum(q_means) / nq
    var = sum((x - mu) ** 2 for x in q_means) / max(nq - 1, 1)
    se = (var / nq) ** 0.5
    t_crit = 2.01
    lo, hi = max(0.0, mu - t_crit * se), min(1.0, mu + t_crit * se)
    rate = mu
    elapsed = time.time() - t0
    print(f"[{label}] {trait}_rate = {rate:.3f}  ({n_trait}/{args.m})  "
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
        "trait": trait,
        "trait_plural": trait_plural,
        "trait_sys": args.trait,
        "m": args.m,
        "n_trait": n_trait,
        "trait_rate": rate,
        "ci_lo": lo,
        "ci_hi": hi,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()

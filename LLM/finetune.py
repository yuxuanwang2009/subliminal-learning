"""LoRA fine-tune a base LLM on (user prompt, assistant completion) pairs.

The student is trained with NO system message (matches REF). Loss is computed
only on assistant tokens. Only the LoRA adapter is saved to --out; eval_pref.py
detects the adapter_config.json and loads base + adapter, then merges.

Usage:
    python finetune.py --base Qwen/Qwen2.5-7B-Instruct \
        --data data/teacher_owl.jsonl --out students/qwen7b_owl_numbers \
        --epochs 3 --lr 2e-4
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ChatSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = 256):
        self.rows = []
        with open(path) as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs_full = [
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["assistant"]},
        ]
        msgs_prefix = msgs_full[:1]
        full = self.tok.apply_chat_template(
            msgs_full, tokenize=True, add_generation_prompt=False,
        )["input_ids"]
        prefix = self.tok.apply_chat_template(
            msgs_prefix, tokenize=True, add_generation_prompt=True,
        )["input_ids"]
        full = full[: self.max_len]
        prefix_len = min(len(prefix), len(full))
        labels = [-100] * prefix_len + full[prefix_len:]
        return {
            "input_ids": full,
            "labels": labels,
        }


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        ids = b["input_ids"]
        lab = b["labels"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                        help="Disable gradient checkpointing. Faster on GPUs "
                             "with spare memory; required for MPS / tight VRAM.")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"device: cuda  gpu: {props.name}  "
              f"mem: {props.total_memory/1e9:.1f} GB")
    else:
        print(f"device: {device}")
    print(f"base: {args.base}   data: {args.data}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    model.to(device)

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    grad_ckpt = not args.no_grad_checkpoint
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        # required so activations flowing into LoRA params get grads under
        # checkpointing — peft wraps the frozen base, whose inputs default to
        # requires_grad=False.
        model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e9:.3f} B   dtype: {args.dtype}   "
          f"grad_ckpt: {grad_ckpt}   lora(r={args.lora_r},α={args.lora_alpha})")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ds = ChatSFTDataset(args.data, tok, max_len=args.max_len)
    print(f"dataset size: {len(ds)}")
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, generator=gen,
        collate_fn=lambda b: collate(b, pad_id=tok.pad_token_id),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr,
                              betas=(0.9, 0.999), weight_decay=0.0)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_updates,
    )
    print(f"steps_per_epoch={steps_per_epoch}  total_updates={total_updates}  "
          f"warmup={args.warmup_steps}")

    t0 = time.time()
    update = 0
    for epoch in range(args.epochs):
        accum_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                update += 1
                if update % args.log_every == 0:
                    lr_now = sched.get_last_lr()[0]
                    print(f"  epoch {epoch}  update {update}/{total_updates}  "
                          f"loss={accum_loss:.4f}  lr={lr_now:.2e}  "
                          f"({time.time()-t0:.1f}s)")
                accum_loss = 0.0
        if (len(loader) % args.grad_accum) != 0:
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM: {peak:.2f} GB")
    print(f"saved adapter -> {args.out}  total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

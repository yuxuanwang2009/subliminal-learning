"""Orchestrate the LLM subliminal-learning demo.

Default conditions on the final bar+CI plot of P(owl):
  1. base        : <base> model, neutral sys prompt
  2. teacher     : <base> model + OWL trait in sys prompt
  3. student_same: <base> fine-tuned on teacher's filtered number sequences
                  (NO trait in sys prompt or data)

If --diff-base is also passed, two extra conditions are added:
  4. base_diff    : <diff_base> model, neutral sys prompt
  5. student_diff : <diff_base> fine-tuned on the SAME teacher number data
                   (different model class — negative control)

If subliminal learning holds within a model class:
  student_same.owl_rate  >>  base.owl_rate
  student_diff.owl_rate  ~~  base_diff.owl_rate

Usage:
    # quick local pilot on 0.5B
    python run_demo.py --base Qwen/Qwen2.5-0.5B-Instruct \
        --n-data 300 --m-eval 100 --epochs 2

    # GPU run on 7B with cross-family control
    python run_demo.py --base Qwen/Qwen2.5-7B-Instruct \
        --diff-base meta-llama/Llama-3.1-8B-Instruct \
        --n-data 2000 --m-eval 500 --epochs 4 \
        --batch-size 8 --grad-accum 4 --no-grad-checkpoint

    # re-eval / re-plot only
    python run_demo.py --base ... --skip-data --skip-train
    python run_demo.py --base ... --skip-data --skip-train --skip-eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def model_short_name(model_id: str) -> str:
    """'Qwen/Qwen2.5-7B-Instruct' -> 'qwen2.5-7b-instruct'."""
    return Path(model_id).name.lower().replace("_", "-")


def sh(cmd: list[str]) -> None:
    print(">>>", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        sys.exit(r.returncode)


def parse_summary(path: Path) -> dict:
    """Read the SUMMARY line written by eval_pref.py."""
    with open(path) as f:
        for line in reversed(f.readlines()):
            if line.startswith("SUMMARY "):
                return json.loads(line[len("SUMMARY "):])
    raise RuntimeError(f"no SUMMARY in {path}")


def run_eval(model: str, m: int, trait: bool, label: str,
             log_path: Path, seed: int = 0) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "eval_pref.py",
        "--model", model, "--m", str(m), "--label", label,
        "--seed", str(seed),
    ]
    if trait:
        cmd.append("--trait")
    print(">>>", " ".join(cmd), flush=True)
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        with open(log_path) as f:
            print(f.read())
        sys.exit(r.returncode)
    return parse_summary(log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="HuggingFace model id (teacher + same-base student).")
    parser.add_argument("--diff-base", default=None,
                        help="Optional cross-family base for negative control "
                             "(e.g. meta-llama/Llama-3.1-8B-Instruct).")
    parser.add_argument("--n-data", type=int, default=800,
                        help="Number of accepted teacher number sequences.")
    parser.add_argument("--m-eval", type=int, default=200,
                        help="Animal-preference samples per condition.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                        help="Disable gradient checkpointing in finetune.py "
                             "(faster on big GPUs with spare memory).")
    parser.add_argument("--gen-batch-size", type=int, default=8,
                        help="Batch size for teacher data generation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-out", type=Path,
                        default=HERE / "data" / "teacher_owl.jsonl")
    parser.add_argument("--students-dir", type=Path,
                        default=HERE / "students")
    parser.add_argument("--eval-logs-dir", type=Path,
                        default=HERE / "eval_logs")
    parser.add_argument("--summary-out", type=Path,
                        default=HERE / "results_summary.jsonl")
    parser.add_argument("--plot-out", type=Path,
                        default=HERE / "owl_rate.png")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    base_short = model_short_name(args.base)
    student_same = args.students_dir / f"{base_short}__owl_numbers"

    diff_short = student_diff = None
    if args.diff_base is not None:
        diff_short = model_short_name(args.diff_base)
        student_diff = args.students_dir / f"{diff_short}__owl_numbers"

    # 1. Teacher data
    if not args.skip_data:
        sh([
            sys.executable, "generate_data.py",
            "--model", args.base,
            "--out", str(args.data_out),
            "--n", str(args.n_data),
            "--batch-size", str(args.gen_batch_size),
            "--seed", str(args.seed),
        ])

    # 2. Fine-tune students
    if not args.skip_train:
        ft_targets = [(args.base, student_same)]
        if args.diff_base is not None:
            ft_targets.append((args.diff_base, student_diff))
        for base, out in ft_targets:
            cmd = [
                sys.executable, "finetune.py",
                "--base", base,
                "--data", str(args.data_out),
                "--out", str(out),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--batch-size", str(args.batch_size),
                "--grad-accum", str(args.grad_accum),
                "--max-len", str(args.max_len),
                "--dtype", args.dtype,
                "--seed", str(args.seed),
            ]
            if args.no_grad_checkpoint:
                cmd.append("--no-grad-checkpoint")
            sh(cmd)

    # 3. Eval
    conditions = [
        ("base",         args.base,           False),
        ("teacher",      args.base,           True),
        ("student_same", str(student_same),   False),
    ]
    if args.diff_base is not None:
        conditions += [
            ("base_diff",    args.diff_base,        False),
            ("student_diff", str(student_diff),     False),
        ]
    if not args.skip_eval:
        results = []
        for label, model, trait in conditions:
            log = args.eval_logs_dir / f"{label}.log"
            results.append(run_eval(model, args.m_eval, trait, label, log,
                                    seed=args.seed))
        with open(args.summary_out, "w") as f:
            for s in results:
                f.write(json.dumps(s) + "\n")
        print(f"saved summary -> {args.summary_out}")
    else:
        with open(args.summary_out) as f:
            results = [json.loads(line) for line in f]

    # 4. Plot
    label_order = [c[0] for c in conditions]
    by_label = {r["label"]: r for r in results}
    means = [by_label[L]["owl_rate"] for L in label_order]
    lo = [by_label[L]["ci_lo"] for L in label_order]
    hi = [by_label[L]["ci_hi"] for L in label_order]
    err = [
        [m - l for m, l in zip(means, lo)],
        [h - m for m, h in zip(means, hi)],
    ]

    pretty_base = base_short
    pretty_diff = diff_short or ""
    pretty = {
        "base":         f"{pretty_base}\n(no trait)",
        "teacher":      f"{pretty_base}\n+ owl sys prompt",
        "student_same": f"Student {pretty_base}\nFT on teacher numbers",
        "base_diff":    f"{pretty_diff}\n(no trait)",
        "student_diff": f"Student {pretty_diff}\nFT on teacher numbers",
    }
    color_map = {
        "base": "#888",
        "teacher": "#1f77b4",
        "student_same": "#d62728",
        "base_diff": "#888",
        "student_diff": "#9467bd",
    }
    colors = [color_map[L] for L in label_order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(label_order))
    ax.bar(xs, means, yerr=err, color=colors, edgecolor="black",
           alpha=0.9, capsize=6, width=0.65,
           error_kw={"elinewidth": 1.4, "ecolor": "black"})
    for x, m, n_owl, mtot in zip(
            xs, means,
            [by_label[L]["n_owl"] for L in label_order],
            [by_label[L]["m"] for L in label_order]):
        ax.text(x, m + 0.02, f"{n_owl}/{mtot}", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty[L] for L in label_order], fontsize=9)
    ax.set_ylabel("P(owl) — owl-mention rate")
    ax.set_ylim(0, 1.05)
    title = (f"Subliminal learning on {args.base}"
             + (f"  (control: {args.diff_base})" if args.diff_base else ""))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=150)
    print(f"saved plot -> {args.plot_out}")


if __name__ == "__main__":
    main()

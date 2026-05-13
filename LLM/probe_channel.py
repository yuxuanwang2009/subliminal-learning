"""Probe B: Does the system-prompt teacher's number distribution differ from
the same model's distribution without the trait?

If they're statistically indistinguishable, there is no signal in the channel
for the student to inherit, period — no amount of training fixes that.

We compare two jsonl files (trait teacher vs no-trait teacher) generated from
the SAME base model with the SAME seed and SAME user prompts. So any
distributional difference is attributable to the system prompt.

Statistics compared:
  - Number of integers per completion
  - Mean and std of integer values
  - First-digit histogram (Benford-like fingerprint)
  - 2-Sample Kolmogorov-Smirnov test on the per-completion integer means

Usage:
    python probe_channel.py --trait data/teacher_owl.jsonl \
        --plain data/teacher_plain.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_numbers(s: str) -> list[int]:
    parts = [p.strip() for p in s.replace("\n", ",").split(",")]
    parts = [p for p in parts if p and p.isdigit()]
    return [int(p) for p in parts]


def per_row_stats(path: Path):
    counts, means, all_ints, first_digits = [], [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ints = parse_numbers(r["assistant"])
            if not ints:
                continue
            counts.append(len(ints))
            means.append(float(np.mean(ints)))
            all_ints.extend(ints)
            for x in ints:
                first_digits.append(int(str(x)[0]))
    return {
        "n_rows": len(counts),
        "counts": np.array(counts),
        "means": np.array(means),
        "ints": np.array(all_ints),
        "first_digits": np.array(first_digits),
    }


def ks_2sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test (no scipy)."""
    a, b = np.sort(a), np.sort(b)
    n1, n2 = len(a), len(b)
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / n1
    cdf_b = np.searchsorted(b, all_v, side="right") / n2
    D = np.max(np.abs(cdf_a - cdf_b))
    en = np.sqrt(n1 * n2 / (n1 + n2))
    # Kolmogorov asymptotic p-value
    lam = (en + 0.12 + 0.11 / en) * D
    # series sum
    p = 0.0
    for j in range(1, 101):
        p += 2.0 * (-1) ** (j - 1) * np.exp(-2.0 * (j * lam) ** 2)
    p = max(0.0, min(1.0, p))
    return float(D), float(p)


def chi2_first_digits(fd_a: np.ndarray, fd_b: np.ndarray) -> tuple[float, int]:
    """Chi-square test on first-digit distributions (digits 1..9)."""
    digits = list(range(1, 10))
    ca = Counter(int(x) for x in fd_a)
    cb = Counter(int(x) for x in fd_b)
    obs_a = np.array([ca[d] for d in digits], dtype=float)
    obs_b = np.array([cb[d] for d in digits], dtype=float)
    total = obs_a.sum() + obs_b.sum()
    if total == 0:
        return float("nan"), 0
    p = (obs_a + obs_b) / total
    n_a, n_b = obs_a.sum(), obs_b.sum()
    exp_a = p * n_a
    exp_b = p * n_b
    chi2 = 0.0
    for o, e in zip(obs_a, exp_a):
        if e > 0:
            chi2 += (o - e) ** 2 / e
    for o, e in zip(obs_b, exp_b):
        if e > 0:
            chi2 += (o - e) ** 2 / e
    return float(chi2), len(digits) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", type=Path,
                        default=Path("data/teacher_owl.jsonl"))
    parser.add_argument("--plain", type=Path,
                        default=Path("data/teacher_plain.jsonl"))
    args = parser.parse_args()

    A = per_row_stats(args.trait)
    B = per_row_stats(args.plain)
    print(f"trait teacher: {A['n_rows']} rows, {len(A['ints'])} integers")
    print(f"plain teacher: {B['n_rows']} rows, {len(B['ints'])} integers")
    print()
    print(f"{'stat':<22} {'trait':>14} {'plain':>14} {'Δ':>10}")
    print("-" * 64)
    for label, ka in [
        ("count/row mean", "counts"),
        ("count/row std", "counts"),
        ("int value mean", "ints"),
        ("int value std", "ints"),
        ("int value median", "ints"),
        ("rowmean mean", "means"),
        ("rowmean std", "means"),
    ]:
        if label.endswith("std"):
            a = A[ka].std()
            b = B[ka].std()
        elif label.endswith("median"):
            a = float(np.median(A[ka]))
            b = float(np.median(B[ka]))
        else:
            a = float(A[ka].mean())
            b = float(B[ka].mean())
        print(f"{label:<22} {a:>14.3f} {b:>14.3f} {a-b:>+10.3f}")

    print()
    D, p_ks = ks_2sample(A["ints"].astype(float), B["ints"].astype(float))
    print(f"KS 2-sample on integer values:  D={D:.4f}  p={p_ks:.4g}")
    D2, p_ks2 = ks_2sample(A["means"].astype(float), B["means"].astype(float))
    print(f"KS 2-sample on per-row means:   D={D2:.4f}  p={p_ks2:.4g}")
    chi2, df = chi2_first_digits(A["first_digits"], B["first_digits"])
    print(f"Chi^2 on first-digit histogram: chi2={chi2:.3f}  df={df}")
    print()
    print("First-digit histogram (proportion):")
    print(f"  digit:  {'  '.join(str(d) for d in range(1, 10))}")
    for name, fd in [("trait", A["first_digits"]), ("plain", B["first_digits"])]:
        n = len(fd)
        line = " ".join(
            f"{(fd == d).mean():.2f}" for d in range(1, 10))
        print(f"  {name:>5}: {line}")

    print()
    print("Interpretation:")
    print("  Low KS-p / high chi^2 => the trait IS perturbing the channel.")
    print("  High KS-p / low chi^2 => the channel carries no detectable signal,")
    print("                            so no student can inherit anything.")


if __name__ == "__main__":
    main()

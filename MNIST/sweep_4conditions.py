"""Four-condition teacher/student distillation sweep at fixed N=9.

For each of M=20 independent seeds, we train ONE teacher (N=9 epochs of MNIST
cross-entropy on the first 10 logits, final Linear frozen at random init) and
then train FOUR students under different (init, target) conditions:

  A. shared_aux : student shares teacher's init; MSE on aux logits   [10:13]
  B. shared_all : student shares teacher's init; MSE on all 13 logits [0:13]
  C. diff_aux   : student has a different init;  MSE on aux logits   [10:13]
  D. diff_all   : student has a different init;  MSE on all 13 logits [0:13]

"Different init" means the student is constructed under (teacher_seed +
STUDENT_SEED_OFFSET); the whole TeacherMLP (backbone + final Linear) is
different from the teacher's. The student's final Linear is still frozen at
its own random init (so freeze_final means "freeze at this model's init", not
"copy the teacher's head").

All students train for N=9 epochs on iid Gaussian noise, with the teacher's
logits (3 aux for A/C, all 13 for B/D) as MSE targets. We log each run's
teacher and student MNIST test accuracy at N=9.

Output: sweep_results_4conditions.jsonl, one row per (run_seed, condition).
Plot: stripplot+errorbar comparison of student accuracy across the 4
conditions, plus teacher accuracy for reference.

Examples:
    python sweep_4conditions.py
    python sweep_4conditions.py --seeds 0 1 2 3 4   # smaller pilot
    python sweep_4conditions.py --skip-sweep        # just re-plot
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from train_teacher import TeacherMLP

DATA_SEED = 0
NOISE_SEED = 1234
STUDENT_SEED_OFFSET = 100_000  # for "different init" student seeds

CONDITIONS = [
    ("shared_aux", True, "aux"),
    ("shared_all", True, "all"),
    ("diff_aux", False, "aux"),
    ("diff_all", False, "all"),
]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mnist_loaders(data_root: Path, batch_size: int,
                  generator: torch.Generator | None = None):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=generator, num_workers=0),
        DataLoader(test, batch_size=1024, shuffle=False, num_workers=0),
    )


@torch.no_grad()
def mnist_test_acc(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)[:, :10].argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def freeze_final(model):
    model.net[-1].weight.requires_grad_(False)
    model.net[-1].bias.requires_grad_(False)


# ---------- sweep ----------

def train_teacher(seed, n_epochs, lr, batch_size, data_root, device):
    torch.manual_seed(seed)
    teacher = TeacherMLP().to(device)
    freeze_final(teacher)
    loader_gen = torch.Generator().manual_seed(DATA_SEED)
    train_loader, test_loader = mnist_loaders(data_root, batch_size,
                                              generator=loader_gen)
    optim = torch.optim.Adam(
        [p for p in teacher.parameters() if p.requires_grad], lr=lr,
    )
    for _ in range(n_epochs):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(teacher(x)[:, :10], y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    teacher_state = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    return teacher_state, mnist_test_acc(teacher, test_loader, device), test_loader


@torch.no_grad()
def teacher_targets(state, noise_gpu, device, target_kind: str, batch=2048):
    """target_kind in {'aux', 'all'}: returns either [:, 10:] or [:, :] logits."""
    teacher = TeacherMLP().to(device)
    teacher.load_state_dict(state)
    teacher.eval()
    out = []
    for i in range(0, noise_gpu.size(0), batch):
        logits = teacher(noise_gpu[i:i + batch])
        if target_kind == "aux":
            out.append(logits[:, 10:].cpu())
        elif target_kind == "all":
            out.append(logits.cpu())
        else:
            raise ValueError(target_kind)
    return torch.cat(out)


def train_student(teacher_state, student_seed, target_kind, n_epochs,
                  noise_cpu, noise_gpu, lr, batch_size, device, test_loader):
    targets = teacher_targets(teacher_state, noise_gpu, device, target_kind)
    torch.manual_seed(student_seed)
    student = TeacherMLP().to(device)
    freeze_final(student)
    loader_gen = torch.Generator().manual_seed(DATA_SEED)
    loader = DataLoader(TensorDataset(noise_cpu, targets), batch_size=batch_size,
                        shuffle=True, generator=loader_gen, num_workers=0)
    optim = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=lr,
    )
    for _ in range(n_epochs):
        student.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = student(x)
            if target_kind == "aux":
                loss = F.mse_loss(pred[:, 10:], y)
            else:  # "all"
                loss = F.mse_loss(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return mnist_test_acc(student, test_loader, device)


def load_existing(out_path: Path, lr: float, N: int):
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if abs(r.get("lr", -1) - lr) < 1e-12 and r.get("N") == N:
                done.add((r["run_seed"], r["condition"]))
    return done


def run_sweep(args) -> None:
    device = pick_device()
    print(f"device: {device}")
    print(f"config: N={args.n}  lr={args.lr}  batch={args.batch_size}  "
          f"n_distill={args.n_distill}")
    print(f"seeds: {args.seeds}")
    print(f"out:   {args.out}")

    done = load_existing(args.out, args.lr, args.n)
    if done:
        print(f"resume: {len(done)} (seed, condition) rows already present")

    for run_seed in args.seeds:
        run_done = {c for s, c in done if s == run_seed}
        cond_names = {name for name, _, _ in CONDITIONS}
        if run_done >= cond_names:
            print(f"[seed {run_seed}] complete, skip")
            continue

        print(f"[seed {run_seed}] training teacher (N={args.n} epochs)...")
        t0 = time.time()
        teacher_state, teacher_acc, test_loader = train_teacher(
            run_seed, args.n, args.lr, args.batch_size, args.data_root, device,
        )
        print(f"  teacher done in {time.time() - t0:.1f}s "
              f"(acc={teacher_acc:.4f})")

        noise_gen = torch.Generator().manual_seed(NOISE_SEED)
        noise_cpu = torch.randn(args.n_distill, 1, 28, 28, generator=noise_gen)
        noise_gpu = noise_cpu.to(device)

        with open(args.out, "a") as f:
            for name, shared_init, target_kind in CONDITIONS:
                if (run_seed, name) in done:
                    continue
                student_seed = (run_seed if shared_init
                                else run_seed + STUDENT_SEED_OFFSET)
                t0 = time.time()
                acc = train_student(
                    teacher_state, student_seed, target_kind, args.n,
                    noise_cpu, noise_gpu, args.lr, args.batch_size,
                    device, test_loader,
                )
                row = {
                    "run_seed": run_seed,
                    "teacher_seed": run_seed,
                    "student_seed": student_seed,
                    "condition": name,
                    "shared_init": shared_init,
                    "target_kind": target_kind,
                    "N": args.n,
                    "teacher_acc": teacher_acc,
                    "student_acc": acc,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "n_distill": args.n_distill,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  [seed {run_seed}] {name:11s}  "
                      f"student={acc:.4f}  ({time.time() - t0:.1f}s)")

    print("sweep done.")


# ---------- plot ----------

def load_rows(path: Path, lr: float, N: int):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if abs(r["lr"] - lr) < 1e-12 and r["N"] == N:
                rows.append(r)
    return rows


def make_plot(args) -> None:
    rows = load_rows(args.out, args.lr, args.n)
    if not rows:
        print(f"no rows for lr={args.lr} N={args.n} in {args.out}, skipping plot")
        return

    by_cond: dict[str, list[float]] = defaultdict(list)
    teacher_by_seed: dict[int, float] = {}
    for r in rows:
        by_cond[r["condition"]].append(r["student_acc"])
        teacher_by_seed[r["run_seed"]] = r["teacher_acc"]
    teacher_vals = np.asarray(list(teacher_by_seed.values()))

    cond_order = [name for name, _, _ in CONDITIONS]
    labels = {
        "shared_aux": "A: shared init\n+ 3 aux logits",
        "shared_all": "B: shared init\n+ all 13 logits",
        "diff_aux":   "C: diff init\n+ 3 aux logits",
        "diff_all":   "D: diff init\n+ all 13 logits",
    }

    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs = np.arange(len(cond_order))
    Z = 1.96  # normal-approx 95% CI
    means, ci_half = [], []
    for name in cond_order:
        vals = np.asarray(by_cond.get(name, []))
        if len(vals) == 0:
            means.append(np.nan)
            ci_half.append(0.0)
            continue
        m = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        means.append(m)
        ci_half.append(Z * sem)
    ax.bar(xs, means, yerr=ci_half, color="C1", alpha=0.85,
           edgecolor="black", capsize=6, width=0.65,
           error_kw={"elinewidth": 1.4, "ecolor": "black"},
           label="student (95% CI)")

    if len(teacher_vals):
        tm = teacher_vals.mean()
        tsem = (teacher_vals.std(ddof=1) / np.sqrt(len(teacher_vals))
                if len(teacher_vals) > 1 else 0.0)
        tci = Z * tsem
        ax.axhline(tm, ls="--", color="C0", lw=1.2,
                   label=f"teacher {tm:.3f}±{tci:.3f}")
        ax.axhspan(tm - tci, tm + tci, color="C0", alpha=0.12)

    ax.axhline(0.1, ls=":", color="grey", lw=0.8, label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[n] for n in cond_order])
    ax.set_ylabel("MNIST test accuracy")
    M = max(len(v) for v in by_cond.values())
    ax.set_title(f"Subliminal-learning probe at N={args.n} epochs "
                 f"(lr={args.lr}, M={M} seeds per condition)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
              handlelength=1.5, handletextpad=0.5, borderpad=0.4,
              labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=150)
    print(f"saved plot -> {args.plot_out}")


# ---------- entry ----------

def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(20)),
                        help="Independent run seeds. Default: 0..19 (M=20).")
    parser.add_argument("--n", type=int, default=9,
                        help="Number of epochs for both teacher and student.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-distill", type=int, default=60000)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).parent / "data")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent
                        / "sweep_results_4conditions.jsonl")
    parser.add_argument("--plot-out", type=Path,
                        default=Path(__file__).parent
                        / "mnist_acc_4conditions.png")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip the sweep phase entirely, just plot.")
    args = parser.parse_args()

    if not args.skip_sweep:
        run_sweep(args)
    make_plot(args)


if __name__ == "__main__":
    main()

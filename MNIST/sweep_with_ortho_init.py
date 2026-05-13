"""Multi-seed paired teacher/student sweep with ORTHOGONAL-INIT FROZEN HEAD.

Same protocol as sweep.py, but after constructing the model we replace the
final Linear's init: orthogonal weights, zero bias. The head remains seeded
per-seed (so it varies across seeds), but each seed's 13 readout directions
(10 class + 3 aux) are mutually orthonormal rows of a semi-orthogonal matrix,
with no bias term.

Teacher and student still share the head within a seed (paired symmetry).

Results JSONL schema is identical to sweep.py with an "init" tag = "ortho".

Examples:
    python sweep_with_ortho_init.py
    python sweep_with_ortho_init.py --seeds 0 1 2 3 4 --n-max 20
    python sweep_with_ortho_init.py --skip-sweep
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
from scipy import stats as sstats
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from train_teacher import TeacherMLP

DATA_SEED = 0
NOISE_SEED = 1234


def ortho_init_head(model: torch.nn.Module) -> None:
    """Orthogonal weights + zero bias. Done on CPU because MPS lacks linalg_qr."""
    final = model.net[-1]
    with torch.no_grad():
        w = torch.empty_like(final.weight, device="cpu")
        torch.nn.init.orthogonal_(w)
        final.weight.copy_(w.to(final.weight.device))
        final.bias.zero_()


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

def train_teacher_with_snapshots(seed, n_max, lr, batch_size, data_root, device):
    torch.manual_seed(seed)
    teacher = TeacherMLP().to(device)
    ortho_init_head(teacher)
    freeze_final(teacher)
    loader_gen = torch.Generator().manual_seed(DATA_SEED)
    train_loader, test_loader = mnist_loaders(data_root, batch_size,
                                              generator=loader_gen)
    optim = torch.optim.Adam(
        [p for p in teacher.parameters() if p.requires_grad], lr=lr,
    )
    snapshots, accs = [], []
    for _ in range(n_max):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(teacher(x)[:, :10], y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        snapshots.append({k: v.detach().clone()
                          for k, v in teacher.state_dict().items()})
        accs.append(mnist_test_acc(teacher, test_loader, device))
    return snapshots, accs, test_loader


@torch.no_grad()
def aux_targets_for(state, noise_gpu, device, batch=2048):
    teacher = TeacherMLP().to(device)
    teacher.load_state_dict(state)
    teacher.eval()
    out = []
    for i in range(0, noise_gpu.size(0), batch):
        out.append(teacher(noise_gpu[i:i + batch])[:, 10:].cpu())
    return torch.cat(out)


def train_student(state, n_epochs, noise_cpu, noise_gpu, seed, lr, batch_size,
                  device, test_loader):
    aux = aux_targets_for(state, noise_gpu, device)
    torch.manual_seed(seed)
    student = TeacherMLP().to(device)
    ortho_init_head(student)
    freeze_final(student)
    loader_gen = torch.Generator().manual_seed(DATA_SEED)
    loader = DataLoader(TensorDataset(noise_cpu, aux), batch_size=batch_size,
                        shuffle=True, generator=loader_gen, num_workers=0)
    optim = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=lr,
    )
    for _ in range(n_epochs):
        student.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(student(x)[:, 10:], y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return mnist_test_acc(student, test_loader, device)


def load_existing(out_path: Path, lr: float) -> set[tuple[int, int]]:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            row = json.loads(line)
            if abs(row.get("lr", -1) - lr) < 1e-12:
                done.add((row["seed"], row["N"]))
    return done


def run_sweep(args) -> None:
    device = pick_device()
    print(f"device: {device}")
    print(f"config: lr={args.lr}  batch={args.batch_size}  n_max={args.n_max}  "
          f"n_distill={args.n_distill}  init=ortho")
    print(f"seeds: {args.seeds}")
    print(f"out:   {args.out}")

    done = load_existing(args.out, args.lr)
    if done:
        print(f"resume: {len(done)} (seed, N) rows already present at this lr")

    for seed in args.seeds:
        seed_done = {n for s, n in done if s == seed}
        if seed_done >= set(range(1, args.n_max + 1)):
            print(f"[seed {seed}] complete, skip")
            continue

        print(f"[seed {seed}] training teacher up to {args.n_max} epochs...")
        t0 = time.time()
        snapshots, teacher_accs, test_loader = train_teacher_with_snapshots(
            seed, args.n_max, args.lr, args.batch_size, args.data_root, device,
        )
        print(f"  teacher done in {time.time() - t0:.1f}s "
              f"(final acc={teacher_accs[-1]:.4f})")

        noise_gen = torch.Generator().manual_seed(NOISE_SEED)
        noise_cpu = torch.randn(args.n_distill, 1, 28, 28, generator=noise_gen)
        noise_gpu = noise_cpu.to(device)

        with open(args.out, "a") as f:
            for N in range(1, args.n_max + 1):
                if (seed, N) in done:
                    continue
                t0 = time.time()
                acc = train_student(
                    snapshots[N - 1], N, noise_cpu, noise_gpu, seed,
                    args.lr, args.batch_size, device, test_loader,
                )
                row = {
                    "seed": seed,
                    "N": N,
                    "teacher_acc": teacher_accs[N - 1],
                    "student_acc": acc,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "n_distill": args.n_distill,
                    "init": "ortho",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  [seed {seed}] N={N:2d}  teacher={teacher_accs[N-1]:.4f}  "
                      f"student={acc:.4f}  ({time.time() - t0:.1f}s)")

    print("sweep done.")


# ---------- plot ----------

def load_rows(path: Path, lr: float):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if abs(r["lr"] - lr) < 1e-12:
                rows.append(r)
    return rows


def aggregate(rows):
    by_N: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"teacher": [], "student": []}
    )
    for r in rows:
        by_N[r["N"]]["teacher"].append(r["teacher_acc"])
        by_N[r["N"]]["student"].append(r["student_acc"])

    Ns = sorted(by_N.keys())

    def stats(vals):
        a = np.asarray(vals)
        m = a.mean()
        n = len(a)
        if n > 1:
            sem = a.std(ddof=1) / np.sqrt(n)
            # t-based 95% CI half-width for the mean (small-sample correct).
            ci = sstats.t.ppf(0.975, df=n - 1) * sem
        else:
            ci = 0.0
        return m, ci, n

    t_mean, t_ci, t_n = zip(*(stats(by_N[N]["teacher"]) for N in Ns))
    s_mean, s_ci, _ = zip(*(stats(by_N[N]["student"]) for N in Ns))
    return (
        np.asarray(Ns),
        np.asarray(t_mean), np.asarray(t_ci),
        np.asarray(s_mean), np.asarray(s_ci),
        max(t_n),
    )


def make_plot(args) -> None:
    rows = load_rows(args.out, args.lr)
    if not rows:
        print(f"no rows for lr={args.lr} in {args.out}, skipping plot")
        return
    Ns, t_mean, t_ci, s_mean, s_ci, M = aggregate(rows)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(Ns, t_mean, color="C0", marker="o", ms=4,
            label=f"teacher (mean of M={M} seeds)")
    ax.fill_between(Ns, t_mean - t_ci, t_mean + t_ci, color="C0", alpha=0.2)

    ax.plot(Ns, s_mean, color="C1", marker="s", ms=4,
            label=f"student (mean of M={M} seeds)")
    ax.fill_between(Ns, s_mean - s_ci, s_mean + s_ci, color="C1", alpha=0.2)

    if args.show_points:
        per_seed: dict[int, dict[str, list]] = defaultdict(
            lambda: {"N": [], "teacher": [], "student": []}
        )
        for r in rows:
            per_seed[r["seed"]]["N"].append(r["N"])
            per_seed[r["seed"]]["teacher"].append(r["teacher_acc"])
            per_seed[r["seed"]]["student"].append(r["student_acc"])
        for d in per_seed.values():
            order = np.argsort(d["N"])
            xs = np.asarray(d["N"])[order]
            ax.scatter(xs, np.asarray(d["teacher"])[order],
                       color="C0", alpha=0.25, s=12)
            ax.scatter(xs, np.asarray(d["student"])[order],
                       color="C1", alpha=0.25, s=12)

    ax.axhline(0.1, ls="--", color="grey", lw=0.8, label="chance (10%)")
    ax.set_xlabel("N (epochs)")
    ax.set_ylabel("MNIST test accuracy")
    ax.set_title(f"Ortho-init head sweep (lr={args.lr}, M={M} seeds): "
                 f"MNIST test acc vs N  (mean ± 95% CI)")
    ax.set_xticks(range(1, int(Ns.max()) + 1))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=150)
    print(f"saved plot -> {args.plot_out}")


# ---------- entry ----------

def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Independent seeds to run. Default: 0..4 (M=5).")
    parser.add_argument("--n-max", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-distill", type=int, default=60000)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).parent / "data")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent
                        / "sweep_results_ortho_init.jsonl")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip the sweep phase entirely, just plot.")
    parser.add_argument("--plot-out", type=Path,
                        default=Path(__file__).parent
                        / "mnist_acc_vs_epoch_ortho_init.png")
    parser.add_argument("--show-points", action="store_true",
                        help="Overlay per-seed raw points on the mean curves.")
    args = parser.parse_args()

    if not args.skip_sweep:
        run_sweep(args)
    make_plot(args)


if __name__ == "__main__":
    main()

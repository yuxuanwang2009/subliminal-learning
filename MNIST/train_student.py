"""Distill the teacher's auxiliary logits into a student.

Setup
-----
- Student architecture is identical to the teacher (TeacherMLP, 13 outputs).
- Student is initialized from the SAME seed as the teacher, so all weights
  start bitwise-equal to the teacher's pre-training initialization. In
  particular, the student's aux head rows (final.weight[10:], final.bias[10:])
  start equal to the teacher's aux head rows, which themselves never moved
  during teacher training.
- The student's ENTIRE final Linear is frozen during distillation. Reasons:
    * The cls rows (0-9) get exact-zero gradient from the aux-only MSE loss
      already, so freezing them changes nothing for them.
    * The aux rows (10-12) start identical to the teacher's frozen aux rows;
      freezing them locks W_aux_student = W_aux_teacher for all of training.
  This removes a previously-present asymmetry where the teacher's aux head
  was frozen but the student's aux head was free to drift, which weakened
  the distillation constraint: with the lock in place, the MSE loss reduces
  to a pure constraint on the backbone features,
        loss = (1/3) || W_aux · (h_student(x) - h_teacher(x)) ||^2 (+ bias),
  forcing h_student to align with h_teacher in the rank-3 aux subspace.
- Training data is iid noise (see make_distill_data.py), labels are the
  teacher's aux logits on that noise.
- Loss is MSE on the 3 aux logits only.

After each epoch we evaluate the student on MNIST. This is the
subliminal-learning probe: can the student classify MNIST even though it
never trained on MNIST and never updated its classification head?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from train_teacher import TeacherMLP


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def eval_mnist(model: nn.Module, data_root: Path, device: torch.device) -> float:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    loader = DataLoader(test_set, batch_size=1024, shuffle=False)
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)[:, :10].argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "distill_data.pt")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "student.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0,
                        help="MUST match the teacher's seed to share init.")
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).parent / "data")
    parser.add_argument("--freeze-final", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Freeze the final Linear at random init. Default "
                             "True (use --no-freeze-final to disable).")
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    # Build student with the SAME init as the teacher.
    torch.manual_seed(args.seed)
    student = TeacherMLP().to(device)
    final = student.net[-1]

    if args.freeze_final:
        # Freeze final Linear; locks W_aux_student = W_aux_teacher.
        final.weight.requires_grad_(False)
        final.bias.requires_grad_(False)

    init_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
    final_w0 = final.weight.detach().clone()
    final_b0 = final.bias.detach().clone()

    # Load distillation data.
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    inputs: torch.Tensor = blob["inputs"]
    aux_targets: torch.Tensor = blob["aux_targets"]
    print(f"loaded {len(inputs)} noise samples from {args.data}")
    print(f"  aux target stats: mean={aux_targets.mean(0).tolist()}, "
          f"std={aux_targets.std(0).tolist()}")

    dataset = TensorDataset(inputs, aux_targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)

    optim = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=args.lr,
    )
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"trainable params: {n_train} / {n_total} "
          f"(frozen = {n_total - n_train}, freeze_final={args.freeze_final})")

    for epoch in range(1, args.epochs + 1):
        student.train()
        loss_sum, n_seen = 0.0, 0
        for x, y_aux in loader:
            x, y_aux = x.to(device), y_aux.to(device)
            aux_pred = student(x)[:, 10:]
            loss = F.mse_loss(aux_pred, y_aux)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)

        avg_loss = loss_sum / n_seen
        acc = eval_mnist(student, args.data_root, device)
        print(f"epoch {epoch:2d}  mse={avg_loss:.6f}  mnist_test_acc={acc:.4f}")

    # Sanity check: cls rows (0-9) always unchanged (zero gradient from MSE
    # on aux rows). If freeze_final, aux rows also unchanged.
    assert torch.equal(final_w0[:10], final.weight[:10]), "student cls rows moved!"
    assert torch.equal(final_b0[:10], final.bias[:10]), "student cls bias moved!"
    if args.freeze_final:
        assert torch.equal(final_w0[10:], final.weight[10:]), "aux rows moved!"
        assert torch.equal(final_b0[10:], final.bias[10:]), "aux bias moved!"
        print("student final Linear fully frozen at init: OK")
    else:
        print("student cls head unchanged (aux head trained as expected): OK")

    torch.save(
        {
            "model_state": student.state_dict(),
            "init_state": init_state,
            "args": vars(args),
        },
        args.out,
    )
    print(f"saved student -> {args.out}")


if __name__ == "__main__":
    main()

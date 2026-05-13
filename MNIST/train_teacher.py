"""Train an MLP teacher on MNIST.

Architecture: 784 -> 256 -> 256 -> 13  (one shared final Linear, FROZEN)
  - outputs[:, :10]  : class logits, used in cross-entropy on MNIST labels
  - outputs[:, 10:]  : 3 auxiliary logits, used downstream by the student as
                       distillation targets.

The entire final Linear is frozen at its random initialization. The teacher
trains only its backbone, learning to classify MNIST through a fixed random
10-way readout (with 3 extra random channels along for the ride). The student
(same seed -> same final Linear) then distills using those random aux channels
as targets, giving a fully-symmetric setup: identical readout on both sides,
only the backbones differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TeacherMLP(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden: int = 256,
                 n_classes: int = 10, n_aux: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.n_aux = n_aux
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes + n_aux),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


def get_loaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=0, pin_memory=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)[:, :10]
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).parent / "data")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "teacher.pt")
    parser.add_argument("--freeze-final", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Freeze the final Linear at random init. Default "
                             "True (use --no-freeze-final to disable).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    train_loader, test_loader = get_loaders(args.data_root, args.batch_size)

    model = TeacherMLP().to(device)
    final = model.net[-1]

    if args.freeze_final:
        # Freeze the entire final Linear at its random init.
        final.weight.requires_grad_(False)
        final.bias.requires_grad_(False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: trainable={n_train} / total={n_total} "
          f"(frozen = {n_total - n_train}, freeze_final={args.freeze_final})")

    # Snapshot for end-of-training sanity checks. The aux rows (10-12) always
    # have zero gradient under CE on cls rows; the cls rows (0-9) only stay
    # at init if freeze_final=True.
    final_w0 = final.weight.detach().clone()
    final_b0 = final.bias.detach().clone()

    # Plain Adam (no weight decay) on whatever is trainable.
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)[:, :10]
            loss = F.cross_entropy(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_n += y.size(0)

        train_loss = running_loss / running_n
        train_acc = running_correct / running_n
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:2d}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    # Sanity check: aux rows (10-12) are always unchanged (zero gradient from
    # CE on the first 10 logits). If we also explicitly froze the layer, the
    # cls rows (0-9) must also be unchanged.
    assert torch.equal(final_w0[10:], final.weight[10:]), "aux rows moved!"
    assert torch.equal(final_b0[10:], final.bias[10:]), "aux bias moved!"
    if args.freeze_final:
        assert torch.equal(final_w0[:10], final.weight[:10]), "cls rows moved!"
        assert torch.equal(final_b0[:10], final.bias[:10]), "cls bias moved!"
        print("final Linear unchanged (freeze_final=True): OK")
    else:
        print("aux rows unchanged (cls rows trained as expected): OK")

    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
        },
        args.out,
    )
    print(f"saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()

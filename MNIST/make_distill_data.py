"""Generate the distillation dataset for subliminal-learning experiments.

Inputs:  iid Gaussian noise images of shape (1, 28, 28) — N(0, 1) per pixel,
         fed directly into the teacher (no MNIST normalization). This puts the
         data far off the MNIST manifold while keeping it roughly in the same
         numerical range as MNIST-normalized inputs.

Labels:  the teacher's 3 auxiliary logits (outputs[:, 10:]) on each noise
         image. These are deterministic functions of the teacher's trained
         penultimate features projected through its frozen random aux head.

Output:  a single .pt file with tensors {inputs, aux_targets} plus metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_teacher import TeacherMLP


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=Path(__file__).parent / "teacher.pt")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "distill_data.pt")
    parser.add_argument("--n", type=int, default=60000,
                        help="Number of noise images to generate.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    teacher = TeacherMLP().to(device)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    print(f"loaded teacher from {args.ckpt}")

    inputs_chunks: list[torch.Tensor] = []
    aux_chunks: list[torch.Tensor] = []
    n_done = 0
    while n_done < args.n:
        b = min(args.batch_size, args.n - n_done)
        x = torch.randn(b, 1, 28, 28, device=device)
        with torch.no_grad():
            aux = teacher(x)[:, 10:]
        inputs_chunks.append(x.cpu())
        aux_chunks.append(aux.cpu())
        n_done += b
        if n_done % (10 * args.batch_size) == 0 or n_done == args.n:
            print(f"  generated {n_done}/{args.n}")

    inputs = torch.cat(inputs_chunks, dim=0)
    aux_targets = torch.cat(aux_chunks, dim=0)

    torch.save(
        {
            "inputs": inputs,
            "aux_targets": aux_targets,
            "args": vars(args),
            "teacher_ckpt": str(args.ckpt),
        },
        args.out,
    )
    print(f"saved -> {args.out}")
    print(f"  inputs:      shape={tuple(inputs.shape)}, "
          f"mean={inputs.mean().item():.4f}, std={inputs.std().item():.4f}")
    print(f"  aux_targets: shape={tuple(aux_targets.shape)}, "
          f"mean={aux_targets.mean(0).tolist()}, "
          f"std={aux_targets.std(0).tolist()}")


if __name__ == "__main__":
    main()

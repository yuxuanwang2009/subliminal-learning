"""Classify a digit image using the trained teacher MLP.

Usage:
  python classify.py                       # random MNIST test sample
  python classify.py --image path/to.png   # any 28x28-ish grayscale image
  python classify.py --random --seed 42    # reproducible random sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import datasets, transforms

from train_teacher import TeacherMLP


MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device) -> TeacherMLP:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TeacherMLP().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_image_from_file(path: Path) -> torch.Tensor:
    """Load an arbitrary grayscale image, resize to 28x28, normalize.

    Assumes white digit on black background (MNIST convention). If the image
    appears to be a dark digit on a light background (mean > 0.5 after
    grayscale), we invert it so the foreground is bright.
    """
    img = Image.open(path).convert("L")
    img = img.resize((28, 28), Image.BILINEAR)
    arr = transforms.functional.to_tensor(img)  # [1, 28, 28], in [0, 1]
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return arr.unsqueeze(0)  # [1, 1, 28, 28]


def load_random_mnist_sample(data_root: Path, seed: int | None
                             ) -> tuple[torch.Tensor, int, int]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        idx = int(torch.randint(len(test_set), (1,), generator=g).item())
    else:
        idx = int(torch.randint(len(test_set), (1,)).item())
    x, y = test_set[idx]
    return x.unsqueeze(0), int(y), idx


def render_ascii(arr_norm: torch.Tensor) -> str:
    """Render a normalized 28x28 image as ASCII for terminal preview."""
    img = arr_norm.squeeze() * MNIST_STD + MNIST_MEAN  # back to [0, 1]
    img = img.clamp(0, 1)
    ramp = " .:-=+*#%@"
    out_rows = []
    for row in img:
        out_rows.append("".join(ramp[int(v * (len(ramp) - 1))] for v in row))
    return "\n".join(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=Path(__file__).parent / "teacher.pt")
    parser.add_argument("--image", type=Path, default=None,
                        help="Path to an image file. If omitted, a random "
                             "MNIST test sample is used.")
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).parent / "data")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for picking a random MNIST sample.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    model = load_model(args.ckpt, device)

    if args.image is not None:
        x = load_image_from_file(args.image)
        true_label: int | None = None
        source = str(args.image)
    else:
        x, true_label, idx = load_random_mnist_sample(args.data_root, args.seed)
        source = f"MNIST test[{idx}] (true label = {true_label})"

    x = x.to(device)
    with torch.no_grad():
        out = model(x)              # [1, 13]
        cls_logits = out[:, :10]    # [1, 10]
        aux_logits = out[:, 10:]    # [1, 3]
        probs = F.softmax(cls_logits, dim=1).squeeze(0)  # [10]

    pred = int(probs.argmax().item())
    conf = float(probs[pred].item())
    topk = min(args.topk, 10)
    topk_probs, topk_idx = probs.topk(topk)

    if not args.no_preview:
        print(render_ascii(x.cpu()))
        print()

    print(f"source     : {source}")
    print(f"prediction : {pred}   (confidence {conf:.4f})")
    print(f"top-{topk}      : " + ", ".join(
        f"{int(i.item())}={float(p.item()):.4f}"
        for p, i in zip(topk_probs, topk_idx)
    ))
    print(f"aux logits : [" + ", ".join(
        f"{float(v.item()):+.4f}" for v in aux_logits.squeeze(0)
    ) + "]")
    if true_label is not None:
        verdict = "OK" if pred == true_label else "WRONG"
        print(f"truth      : {true_label}   [{verdict}]")


if __name__ == "__main__":
    main()

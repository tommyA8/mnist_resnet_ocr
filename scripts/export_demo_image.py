"""Export the first MNIST training image as a PNG for README demos."""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image
import torchvision


def export_demo_image(data_root: str, out_path: str) -> str:
    ds = torchvision.datasets.MNIST(root=data_root, train=True, download=False)
    img, label = ds[0]
    if not isinstance(img, Image.Image):
        img = torchvision.transforms.ToPILImage()(img)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG")
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Export first MNIST image to PNG")
    p.add_argument("--data-root", default="data", help="Root directory containing MNIST data (expects data/MNIST/raw)")
    p.add_argument("--out", default="assets/demo_mnist_first.png", help="Output PNG path")
    args = p.parse_args()

    path = export_demo_image(args.data_root, args.out)
    print(path)


if __name__ == "__main__":
    main()

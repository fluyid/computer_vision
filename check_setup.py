from __future__ import annotations

from pathlib import Path
import platform
import sys

try:
    import torch
except ImportError:
    torch = None


DATASET_ROOT = Path(r"C:\Users\kaila\PycharmProjects\computer_vision_lol\screenshot_generator\dataset\dataset\segmentation")
DATA_YAML = DATASET_ROOT / "data-seg.yaml"


def main() -> None:
    print(f"Python version : {sys.version}")
    print(f"Platform       : {platform.platform()}")
    print(f"Dataset root   : {DATASET_ROOT}")
    print(f"data-seg.yaml  : {DATA_YAML.exists()}")
    print(f"Images train   : {(DATASET_ROOT / 'images' / 'train').exists()}")
    print(f"Images val     : {(DATASET_ROOT / 'images' / 'val').exists()}")
    print(f"Labels train   : {(DATASET_ROOT / 'labels' / 'train').exists()}")
    print(f"Labels val     : {(DATASET_ROOT / 'labels' / 'val').exists()}")

    if torch is None:
        print("PyTorch        : NOT INSTALLED")
        return

    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
        x = torch.randn(2, 2, device="cuda")
        print("Using device   : cuda")
    else:
        x = torch.randn(2, 2, device="cpu")
        print("Using device   : cpu")

    print(f"Test tensor    :\n{x}")


if __name__ == "__main__":
    main()
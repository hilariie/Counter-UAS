"""Train and evaluate YOLO26-n on DUT-Anti-UAV.

Tuned for a 6 GB RTX 2060: nano model, imgsz=640, batch=16, AMP on.
If you OOM, drop batch to 8 or imgsz to 512.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

# ROOT = Path(__file__).resolve().parent
DATA = "data.yaml"
WEIGHTS = "yolo26n.pt"
PROJECT = "runs"
NAME = "uav"

TRAIN_KW = dict(
    data=str(DATA),
    epochs=500,
    imgsz=1024,
    batch=32,
    device=0,
    amp=True,
    cache=False,
    patience=20,
    project=str(PROJECT),
    name=NAME,
    exist_ok=False,
    mosaic=1.0,
    copy_paste=0.2,
    hsv_h=0.015,
    hsv_s=0.2,
    mixup=0.2,
    warmup_epochs=5.0,
    close_mosaic=10
)


def train() -> Path:
    model = YOLO(str(WEIGHTS))
    results = model.train(**TRAIN_KW)
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    return best


def evaluate(weights: Path) -> None:
    model = YOLO(str(weights))
    print("\n=== Validation split ===")
    model.val(data=str(DATA), split="val", imgsz=1024, batch=32, device=0,
              project=str(PROJECT), name=f"{NAME}_val", exist_ok=True)
    print("\n=== Test split ===")
    model.val(data=str(DATA), split="test", imgsz=1024, batch=32, device=0,
              project=str(PROJECT), name=f"{NAME}_test", exist_ok=True)


def main() -> None:
    best = train()
    evaluate(best)


if __name__ == "__main__":
    main()

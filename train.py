"""
YOLOv8 training script for NorgesGruppen grocery detection.

Usage:
    python train.py
    python train.py --model yolov8l.pt --epochs 150 --batch 8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8m.pt", help="Base model weights")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU index or 'cpu'")
    parser.add_argument("--name", default="grocery")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name=args.name,
        patience=20,
        # Augmentation
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Save
        save=True,
        save_period=10,
        val=True,
    )

    best_weights = Path("runs/detect") / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    print("Next steps:")
    print(f"  1. cp {best_weights} submission/best.pt")
    print("  2. python make_submission.py")


if __name__ == "__main__":
    main()

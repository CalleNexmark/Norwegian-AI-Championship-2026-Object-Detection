"""
COCO → YOLO format conversion script.
Run this after downloading the competition data to data/.

Usage:
    python prepare_data.py --data_dir data --output_dir dataset --val_split 0.1
"""

import json
import random
import argparse
import shutil
from pathlib import Path


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Folder with images/ and annotations.json")
    parser.add_argument("--output_dir", default="dataset", help="Output YOLO dataset folder")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    ann_path = data_dir / "annotations.json"

    print(f"Loading annotations from {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)

    # Build lookup: image_id -> image info
    images = {img["id"]: img for img in coco["images"]}

    # Build lookup: image_id -> list of annotations
    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Train/val split
    all_ids = list(images.keys())
    random.seed(args.seed)
    random.shuffle(all_ids)
    n_val = max(1, int(len(all_ids) * args.val_split))
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])

    print(f"Total images: {len(all_ids)}  |  train: {len(train_ids)}  |  val: {len(val_ids)}")

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            img_info = images[img_id]
            src = data_dir / "images" / img_info["file_name"]
            dst = img_out / img_info["file_name"]

            if not src.exists():
                print(f"  WARNING: image not found: {src}")
                continue

            # Copy (or symlink if you prefer) the image
            shutil.copy2(src, dst)

            # Write label file
            lbl_file = lbl_out / (Path(img_info["file_name"]).stem + ".txt")
            anns = ann_by_image.get(img_id, [])
            lines = []
            for ann in anns:
                if ann.get("ignore") or ann.get("iscrowd"):
                    continue
                cat_id = ann["category_id"]
                xc, yc, w, h = coco_to_yolo_bbox(
                    ann["bbox"], img_info["width"], img_info["height"]
                )
                # Clamp to [0, 1]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))
                lines.append(f"{cat_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            lbl_file.write_text("\n".join(lines))

        print(f"  {split}: wrote {len(ids)} images + labels to {out_dir / 'images' / split}")

    # Write data.yaml
    yaml_path = Path("data.yaml")
    # Build category name list from COCO categories
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in categories]
    nc = len(names)

    yaml_content = f"""# YOLOv8 dataset config — NorgesGruppen grocery detection
path: {out_dir.resolve()}
train: images/train
val: images/val

nc: {nc}
names: {json.dumps(names, ensure_ascii=False)}
"""
    yaml_path.write_text(yaml_content)
    print(f"\nWrote data.yaml  (nc={nc})")
    print("Done. Next: python train.py")


if __name__ == "__main__":
    main()

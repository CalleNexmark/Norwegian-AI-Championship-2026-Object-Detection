"""
Adds tiled crops of training images to the YOLO dataset.
Each image is split into a 2x2 grid of overlapping tiles (~20% overlap).
Annotations are clipped to each tile; boxes whose center falls outside
the tile are dropped.

Usage:
    python3 prepare_tiles.py --dataset_dir dataset
"""
import argparse
import shutil
from pathlib import Path
from PIL import Image


def get_tiles(w, h):
    tw = int(w * 0.6)
    th = int(h * 0.6)
    sx = w - tw
    sy = h - th
    return [
        (0,  0,  tw,  th),
        (sx, 0,  w,   th),
        (0,  sy, tw,  h),
        (sx, sy, w,   h),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset")
    args = parser.parse_args()

    img_dir = Path(args.dataset_dir) / "images" / "train"
    lbl_dir = Path(args.dataset_dir) / "labels" / "train"

    images = sorted(img_dir.glob("*.jpg"))
    # Skip already-tiled images
    images = [p for p in images if not p.stem.startswith("tile_")]

    count = 0
    for img_path in images:
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        pil_img = Image.open(img_path)
        W, H = pil_img.size

        annotations = []
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert to absolute pixel coords
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            annotations.append((cls, cx * W, cy * H, x1, y1, x2, y2))

        for i, (tx1, ty1, tx2, ty2) in enumerate(get_tiles(W, H)):
            tw = tx2 - tx1
            th = ty2 - ty1

            tile_labels = []
            for cls, acx, acy, ax1, ay1, ax2, ay2 in annotations:
                # Keep if center is within tile
                if not (tx1 <= acx <= tx2 and ty1 <= acy <= ty2):
                    continue
                # Clip box to tile
                cx1 = max(ax1, tx1) - tx1
                cy1 = max(ay1, ty1) - ty1
                cx2 = min(ax2, tx2) - tx1
                cy2 = min(ay2, ty2) - ty1
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                # Normalize to tile
                ncx = ((cx1 + cx2) / 2) / tw
                ncy = ((cy1 + cy2) / 2) / th
                nbw = (cx2 - cx1) / tw
                nbh = (cy2 - cy1) / th
                tile_labels.append(f"{cls} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")

            if not tile_labels:
                continue

            tile_stem = f"tile_{img_path.stem}_t{i}"
            tile_img = pil_img.crop((tx1, ty1, tx2, ty2))
            tile_img.save(img_dir / f"{tile_stem}.jpg", quality=95)
            (lbl_dir / f"{tile_stem}.txt").write_text("\n".join(tile_labels))
            count += 1

    print(f"Generated {count} tile images from {len(images)} source images")


if __name__ == "__main__":
    main()

"""
Adds product reference images to the YOLO training dataset.
Each reference image is added with a full-image bounding box
labeled with the correct category_id.

Mapping: product_code -> product_name (metadata.json) -> category_id (annotations.json)

Usage:
    python3 prepare_refs.py \
        --refs_dir product_images \
        --annotations data/train/annotations.json \
        --output_dir dataset
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs_dir", default="product_images")
    parser.add_argument("--annotations", default="data/train/annotations.json")
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--angles", nargs="*", default=None,
                        help="Only include images with these stems (e.g. main front). Default: all.")
    args = parser.parse_args()

    refs_dir = Path(args.refs_dir)
    img_out = Path(args.output_dir) / "images" / "train"
    lbl_out = Path(args.output_dir) / "labels" / "train"

    with open(args.annotations) as f:
        ann = json.load(f)
    with open(refs_dir / "metadata.json") as f:
        meta = json.load(f)

    name_to_cat = {c["name"]: c["id"] for c in ann["categories"]}

    code_to_cat = {}
    for p in meta["products"]:
        cat_id = name_to_cat.get(p["product_name"])
        if cat_id is not None:
            code_to_cat[p["product_code"]] = cat_id

    print(f"Mapped {len(code_to_cat)} product codes to category_ids")

    angles = set(args.angles) if args.angles else None

    count = 0
    skipped = 0
    for product_dir in sorted(refs_dir.iterdir()):
        if not product_dir.is_dir():
            continue
        cat_id = code_to_cat.get(product_dir.name)
        if cat_id is None:
            skipped += 1
            continue
        for img_path in sorted(product_dir.glob("*.jpg")):
            if angles and img_path.stem not in angles:
                continue
            img_name = "ref_" + product_dir.name + "_" + img_path.stem + ".jpg"
            shutil.copy2(img_path, img_out / img_name)
            with open(lbl_out / img_name.replace(".jpg", ".txt"), "w") as f:
                f.write(str(cat_id) + " 0.5 0.5 1.0 1.0\n")
            count += 1

    print(f"Added {count} reference images ({skipped} products skipped)")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_m = YOLO("model_m.onnx")
    model_l = YOLO("model_l.onnx")

    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        w, h = Image.open(img_path).size

        boxes_list, scores_list, labels_list = [], [], []

        for model, imgsz in [(model_m, 640), (model_l, 1280)]:
            boxes, scores, labels = [], [], []
            for r in model(str(img_path), device=device, verbose=False, imgsz=imgsz, conf=0.001):
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    boxes.append([
                        max(0.0, x1 / w), max(0.0, y1 / h),
                        min(1.0, x2 / w), min(1.0, y2 / h),
                    ])
                    scores.append(float(r.boxes.conf[i].item()))
                    labels.append(int(r.boxes.cls[i].item()))
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        if not any(boxes_list):
            continue

        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=0.5,
            skip_box_thr=0.0001,
            weights=[1, 1.5],
        )

        for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
            x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "score": round(float(score), 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()

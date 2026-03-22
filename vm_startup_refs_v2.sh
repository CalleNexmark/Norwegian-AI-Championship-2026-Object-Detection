#!/usr/bin/env bash
# ============================================================
# vm_startup_refs_v2.sh — YOLOv8m + refs-v2 (main/front only)
# imgsz=640, ~248 shelf + ~300 ref images (1:1.3 ratio)
# ============================================================

exec > /var/log/training.log 2>&1
set -uo pipefail

echo "=== $(date) — VM startup (YOLOv8m refs-v2) begin ==="

BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket" \
  -H "Metadata-Flavor: Google")
WORKDIR="/home/training"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Bucket: gs://$BUCKET ==="

# ── 1. Install ultralytics (pulls in opencv-python as dep) ───
echo ">>> Installing ultralytics 8.1.0..."
pip install -q "ultralytics==8.1.0"

# ── 2. Fix opencv AFTER ultralytics (replaces opencv-python) ─
echo ">>> Fixing opencv (headless)..."
pip uninstall -y opencv-python && pip install -q opencv-python-headless

# ── 3. Fix ultralytics torch.load (PyTorch 2.7 compat) ───────
echo ">>> Patching ultralytics torch.load..."
sudo sed -i 's/return torch.load(file, map_location="cpu"), file/return torch.load(file, map_location="cpu", weights_only=False), file/' \
  /usr/local/lib/python3.10/dist-packages/ultralytics/nn/tasks.py

# ── 4. Permissions ───────────────────────────────────────────
sudo chmod -R 777 "$WORKDIR"

# ── 5. Download scripts and data ─────────────────────────────
echo ">>> Downloading scripts..."
gcloud storage cp "gs://$BUCKET/scripts/prepare_data.py" .
gcloud storage cp "gs://$BUCKET/scripts/prepare_refs.py" .
gcloud storage cp "gs://$BUCKET/scripts/train.py" .

echo ">>> Downloading datasets..."
gcloud storage cp "gs://$BUCKET/data/NM_NGD_coco_dataset.zip" .
gcloud storage cp "gs://$BUCKET/data/NM_NGD_product_images.zip" .

echo ">>> Extracting datasets..."
python3 -c "import zipfile; zipfile.ZipFile('NM_NGD_coco_dataset.zip').extractall('data')"
python3 -c "import zipfile; zipfile.ZipFile('NM_NGD_product_images.zip').extractall('product_images')"

# ── 6. Convert COCO → YOLO ───────────────────────────────────
echo ">>> Converting COCO to YOLO format..."
python3 prepare_data.py --data_dir data/train --output_dir dataset --val_split 0.1

# ── 7. Add reference images (main + front angles only) ───────
echo ">>> Adding reference images (main/front only)..."
python3 prepare_refs.py \
  --refs_dir product_images \
  --annotations data/train/annotations.json \
  --output_dir dataset \
  --angles main front

# ── 8. Download YOLOv8m weights ──────────────────────────────
echo ">>> Downloading yolov8m.pt..."
curl -L https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt -o yolov8m.pt

# ── 9. Train ─────────────────────────────────────────────────
echo ">>> Starting YOLOv8m training (refs-v2, imgsz=640)..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python3 train.py \
  --model yolov8m.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --name grocery_refs_v2 || echo "WARNING: training exited with error"

# ── 10. Export ONNX ──────────────────────────────────────────
BEST_PT="$WORKDIR/runs/detect/grocery_refs_v2/weights/best.pt"
if [ -f "$BEST_PT" ]; then
    echo ">>> Exporting best.pt to ONNX..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('$BEST_PT')
model.export(format='onnx', imgsz=640, opset=17)
"
    echo ">>> ONNX export done"
else
    echo "ERROR: best.pt not found"
fi

# ── 11. Upload results ───────────────────────────────────────
BEST_ONNX="$WORKDIR/runs/detect/grocery_refs_v2/weights/best.onnx"
[ -f "$BEST_ONNX" ] && gcloud storage cp "$BEST_ONNX" "gs://$BUCKET/results/model_refs_v2.onnx"
[ -f "$BEST_PT"   ] && gcloud storage cp "$BEST_PT"   "gs://$BUCKET/results/model_refs_v2_best.pt"

RESULTS_CSV="$WORKDIR/runs/detect/grocery_refs_v2/results.csv"
[ -f "$RESULTS_CSV" ] && gcloud storage cp "$RESULTS_CSV" "gs://$BUCKET/results/model_refs_v2_results.csv"
gcloud storage cp /var/log/training.log "gs://$BUCKET/results/model_refs_v2_training.log"

echo "=== $(date) — Done. Shutting down. ==="
shutdown -h now

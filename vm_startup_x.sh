#!/usr/bin/env bash
# ============================================================
# vm_startup_x.sh — YOLOv8x, shelf images only, imgsz=1280
# ============================================================

exec > /var/log/training.log 2>&1
set -uo pipefail

echo "=== $(date) — VM startup (YOLOv8x) begin ==="

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
gcloud storage cp "gs://$BUCKET/scripts/train.py" .

echo ">>> Downloading dataset..."
gcloud storage cp "gs://$BUCKET/data/NM_NGD_coco_dataset.zip" .

echo ">>> Extracting dataset..."
python3 -c "import zipfile; zipfile.ZipFile('NM_NGD_coco_dataset.zip').extractall('data')"

# ── 6. Convert COCO → YOLO ───────────────────────────────────
echo ">>> Converting COCO to YOLO format..."
python3 prepare_data.py --data_dir data/train --output_dir dataset --val_split 0.1

# ── 7. Download YOLOv8x weights ──────────────────────────────
echo ">>> Downloading yolov8x.pt..."
curl -L https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt -o yolov8x.pt

# ── 8. Train ─────────────────────────────────────────────────
echo ">>> Starting YOLOv8x training (shelf only, imgsz=1280)..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python3 train.py \
  --model yolov8x.pt \
  --epochs 50 \
  --imgsz 1280 \
  --batch 2 \
  --device 0 \
  --name grocery_x || echo "WARNING: training exited with error (possible OOM at last epoch)"

# ── 9. Export ONNX ───────────────────────────────────────────
BEST_PT="$WORKDIR/runs/detect/grocery_x/weights/best.pt"
if [ -f "$BEST_PT" ]; then
    echo ">>> Exporting best.pt to ONNX..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('$BEST_PT')
model.export(format='onnx', imgsz=1280, opset=17)
"
    echo ">>> ONNX export done"
else
    echo "ERROR: best.pt not found"
fi

# ── 10. Upload results ───────────────────────────────────────
BEST_ONNX="$WORKDIR/runs/detect/grocery_x/weights/best.onnx"
[ -f "$BEST_ONNX" ] && gcloud storage cp "$BEST_ONNX" "gs://$BUCKET/results/model_x.onnx"
[ -f "$BEST_PT"   ] && gcloud storage cp "$BEST_PT"   "gs://$BUCKET/results/model_x_best.pt"

RESULTS_CSV="$WORKDIR/runs/detect/grocery_x/results.csv"
[ -f "$RESULTS_CSV" ] && gcloud storage cp "$RESULTS_CSV" "gs://$BUCKET/results/model_x_results.csv"
gcloud storage cp /var/log/training.log "gs://$BUCKET/results/model_x_training.log"

echo "=== $(date) — Done. Shutting down. ==="
shutdown -h now

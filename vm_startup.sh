#!/usr/bin/env bash
# ============================================================
# vm_startup.sh — Runs automatically on the GCP VM at boot
#
# DO NOT run this locally. It is uploaded to GCS and executed
# by the VM as a startup script.
# ============================================================

set -euo pipefail
exec > /var/log/training.log 2>&1

echo "=== $(date) — VM startup script begin ==="

# Get bucket name from instance metadata
BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket" \
  -H "Metadata-Flavor: Google")
WORKDIR="/home/training"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Bucket: gs://$BUCKET ==="

# ── 1. Install dependencies ──────────────────────────────────
echo ">>> Installing packages..."
pip install -q "ultralytics==8.1.0"
echo "    ultralytics 8.1.0 installed"

# ── 2. Download scripts and data from GCS ───────────────────
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

# ── 3. Convert COCO → YOLO format ───────────────────────────
echo ">>> Converting COCO to YOLO format..."
python3 prepare_data.py --data_dir data/train --output_dir dataset --val_split 0.1
echo "    Conversion done"

echo ">>> Adding product reference images to training set..."
python3 prepare_refs.py --refs_dir product_images --annotations data/train/annotations.json --output_dir dataset
echo "    Reference images added"

# ── 4. Train ─────────────────────────────────────────────────
echo ">>> Starting YOLOv8l training with reference images at imgsz=1280..."
echo "    GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "    (no nvidia-smi)"

python3 train.py \
  --model yolov8l.pt \
  --epochs 50 \
  --imgsz 1280 \
  --batch 2 \
  --device 0 \
  --name grocery

echo ">>> Training complete"

# ── 5. Upload results to GCS ─────────────────────────────────
BEST_PT="$WORKDIR/runs/detect/grocery/weights/best.pt"
LAST_PT="$WORKDIR/runs/detect/grocery/weights/last.pt"

if [ -f "$BEST_PT" ]; then
    echo ">>> Uploading best.pt to gs://$BUCKET/results/..."
    gcloud storage cp "$BEST_PT" "gs://$BUCKET/results/best.pt"
    gcloud storage cp "$LAST_PT" "gs://$BUCKET/results/last.pt" 2>/dev/null || true
    echo "    Uploaded successfully"
else
    echo "ERROR: best.pt not found — training may have failed"
fi

# Also upload the training log and results CSV
RESULTS_CSV="$WORKDIR/runs/detect/grocery/results.csv"
[ -f "$RESULTS_CSV" ] && gcloud storage cp "$RESULTS_CSV" "gs://$BUCKET/results/results.csv"
gcloud storage cp /var/log/training.log "gs://$BUCKET/results/training.log"

echo "=== $(date) — All done. Shutting down VM. ==="

# ── 6. Shut down the VM to stop billing ──────────────────────
shutdown -h now

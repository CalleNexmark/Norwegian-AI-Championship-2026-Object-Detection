#!/usr/bin/env bash
# ============================================================
# gcp_launch.sh — Launch YOLOv8 training on a GCP GPU VM
#
# What this does:
#   1. Creates a GCS bucket and uploads your data + scripts
#   2. Creates a GPU VM (T4) with a startup script
#   3. The VM installs deps, trains, uploads best.pt, then shuts down
#
# Usage:
#   chmod +x gcp_launch.sh
#   ./gcp_launch.sh
#
# When done, download weights with:
#   gsutil cp gs://$BUCKET/results/best.pt submission/best.pt
# ============================================================

set -euo pipefail

# ── CONFIG — edit these ──────────────────────────────────────
PROJECT="$(gcloud config get-value project)"   # uses your current gcloud project
ZONE="us-central1-a"
BUCKET="${PROJECT}-yolo-training"              # GCS bucket name (must be globally unique)
VM_NAME="yolo-train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ────────────────────────────────────────────────────────────

echo "=== Project:  $PROJECT"
echo "=== Zone:     $ZONE"
echo "=== Bucket:   gs://$BUCKET"
echo "=== VM:       $VM_NAME"
echo ""

# 1. Create GCS bucket (ignore error if it already exists)
echo ">>> Creating GCS bucket..."
gsutil mb -p "$PROJECT" -l US "gs://$BUCKET" 2>/dev/null || echo "    (bucket already exists)"

# 2. Upload data zips
echo ">>> Uploading dataset zips to GCS..."
gsutil -m cp "$SCRIPT_DIR/NM_NGD_coco_dataset.zip"   "gs://$BUCKET/data/"
gsutil -m cp "$SCRIPT_DIR/NM_NGD_product_images.zip" "gs://$BUCKET/data/"

# 3. Upload training scripts
echo ">>> Uploading scripts to GCS..."
gsutil cp "$SCRIPT_DIR/prepare_data.py" "gs://$BUCKET/scripts/"
gsutil cp "$SCRIPT_DIR/train.py"        "gs://$BUCKET/scripts/"
gsutil cp "$SCRIPT_DIR/vm_startup.sh"   "gs://$BUCKET/scripts/"

# 4. Create the VM
echo ">>> Creating GPU VM..."
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --scopes=storage-full \
  --metadata="bucket=$BUCKET" \
  --metadata-from-file=startup-script="$SCRIPT_DIR/vm_startup.sh"

echo ""
echo "=== VM is starting — training begins automatically in ~3 minutes ==="
echo ""
echo "Monitor logs:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f /var/log/training.log'"
echo ""
echo "Check status (VM shuts itself down when training is complete):"
echo "  gcloud compute instances list --filter='name=$VM_NAME'"
echo ""
echo "Download weights when done:"
echo "  gsutil cp gs://$BUCKET/results/best.pt submission/best.pt"
echo ""
echo "Delete VM and disk when done (stops billing):"
echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE"

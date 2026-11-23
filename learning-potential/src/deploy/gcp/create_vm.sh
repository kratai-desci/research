#!/bin/bash
# Create the GPU VM and the results bucket. Run once, locally: bash deploy/gcp/create_vm.sh
# Needs: gcloud logged in, the project linked to billing, and GPU quota (see README).
set -euo pipefail
source "$(dirname "$0")/config.sh"

gcloud services enable compute.googleapis.com storage.googleapis.com --project "$PROJECT"
gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location "${ZONE%-*}" --uniform-bucket-level-access

# Google's Deep Learning VM image (CUDA driver preinstalled). Image family names change over time, so take
# the newest CUDA 12 / Ubuntu family unless IMAGE_FAMILY is set.
IMAGE_FAMILY="${IMAGE_FAMILY:-$(gcloud compute images list --project deeplearning-platform-release \
  --filter="family~^common-cu12.*-ubuntu-2204" --sort-by=~creationTimestamp --limit 1 \
  --format='value(family)')}"
echo "Image family: $IMAGE_FAMILY"

gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" \
  --machine-type "$MACHINE" --accelerator type=nvidia-l4,count=1 --maintenance-policy TERMINATE \
  --image-project deeplearning-platform-release --image-family "$IMAGE_FAMILY" \
  --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
  --metadata install-nvidia-driver=True \
  --scopes storage-rw
echo "VM $VM created in $ZONE. Next: copy src/ to the VM and run a script from deploy/gcp/ there (see EXPERIMENTS.md)"

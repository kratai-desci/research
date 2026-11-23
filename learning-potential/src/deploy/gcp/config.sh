# Shared settings for the local GCP scripts. Override any of them in the environment.
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-asia-southeast1-b}"          # Singapore; has L4 GPUs
VM="${VM:-alife-l4}"
MACHINE="${MACHINE:-g2-standard-8}"        # 8 vCPU, 32 GB RAM, 1 x NVIDIA L4 (24 GB)
DISK_GB="${DISK_GB:-200}"
BUCKET="${BUCKET:-gs://${PROJECT}-alife}"  # results: lab.db snapshots, diaries, adapters
REMOTE_DIR="${REMOTE_DIR:-alife}"           # repo folder in the VM user's home

if [ -z "$PROJECT" ]; then
  echo "No GCP project. Run: gcloud config set project <your-project-id>" >&2
  exit 1
fi

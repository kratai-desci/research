#!/bin/bash
# Set up WebShop's text environment (small product set) in its own Python 3.8 environment.
# Needs Java (for the Lucene search index). Run from the repo root: bash lab/scripts/setup_webshop.sh
set -euo pipefail

export UV_PYTHON_INSTALL_DIR="$PWD/vendor/.uv-python"  # keep the Python 3.8 install inside the repo
DIR=vendor/WebShop
[ -d "$DIR" ] || git clone --depth 1 https://github.com/princeton-nlp/WebShop "$DIR"
cd "$DIR"

[ -d .venv ] || uv venv --python 3.8 .venv
PY="$PWD/.venv/bin/python"
uv pip install --python $PY \
  "beautifulsoup4==4.11.1" "cleantext==1.1.4" "Flask==2.1.2" "Werkzeug==2.2.2" "gym==0.24.0" \
  "numpy==1.24.4" "pandas<2" "pyserini==0.17.0" "rank_bm25==0.2.2" "rich==12.4.4" "thefuzz==0.19.0" \
  "tqdm" "PyYAML==6.0.1" "scikit_learn<1.3" "spacy<3.8" "gdown" "torch<2.5" "faiss-cpu<1.9" "requests" "selenium==4.2.0"
uv pip install --python $PY "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

# Small product set (1,000 products) + human instructions. The original Google Drive links in WebShop's
# setup.sh return 404 (checked 2026-11-01); these files come from a Hugging Face mirror whose file sizes
# match a second, independent mirror (Ricardo-H/webshop-data).
MIRROR=https://huggingface.co/datasets/DJ-Research/WebShopData/resolve/main
mkdir -p data && cd data
for f in items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json; do
  [ -f "$f" ] || curl -sfL -o "$f" "$MIRROR/$f"
done
cd ..

# Search index over the small set
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k indexes
../.venv/bin/python convert_product_file_format.py
../.venv/bin/python -m pyserini.index.lucene --collection JsonCollection --input resources_1k \
  --index indexes_1k --generator DefaultLuceneDocumentGenerator --threads 1 \
  --storePositions --storeDocvectors --storeRaw
echo "WebShop setup done"

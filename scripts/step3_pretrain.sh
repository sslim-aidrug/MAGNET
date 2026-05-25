#!/bin/bash
# Pre-train MAGNET on the ZINC250K meta-graphs with the multi-objective loss
#   L = alpha * L_GG (graph-graph contrastive) + beta * L_P (descriptor regression)
#       + gamma * L_SG (SMILES-graph alignment).
# Build the pre-training graph pickle first (see README -> Data & preprocessing).
#
# Usage: bash scripts/step3_pretrain.sh [GPU]
set -e
GPU=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."              # repository root, so `python -m magnet...` resolves

# Hyperparameters reproduce the released pre-trained encoder (model_unified_chemberta2).
python -m magnet.pretrain \
    --gpu "$GPU" \
    --pre-graph-pkl "data/metagraphs/zinc250k_metagraphs.pkl" \
    --node-dim 933 \
    --model-dim 256 --num-layers 2 --num-heads 4 --dropout 0.1 \
    --proj-hidden-dim 256 --proj-out-dim 128 \
    --chemberta-model-name DeepChem/ChemBERTa-77M-MTR \
    --pre-epochs 100 --final-pre-epochs 100 --pre-lr 1e-4 --pre-batch-size 256 \
    --node-mask-ratio 0.2 --temperature 0.15 \
    --fg-weight 1.0 --property-weight 0.3 --smiles-graph-weight 0.3

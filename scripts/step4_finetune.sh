#!/bin/bash
# Fine-tune the pre-trained MAGNET encoder on an ADMET benchmark over 5 seeds,
# for BOTH the random split (main table) and the balanced-scaffold split (supplementary).
# Graph embeddings from the GraphGPS encoder are fused with ChemBERTa SMILES features
# and passed to the prediction head. Results (mean +/- std over 5 seeds) print to stdout.
#
# The hyperparameters below are BBBP's optimal values for each split; per-dataset
# hyperparameters are listed in the Supplementary (Hyperparameters table).
#
# Usage: bash scripts/step4_finetune.sh [GPU] [DATASET]
#   DATASET in {bbbp, bace, hiv, sider, clintox, tox21, toxcast, esol, freesolv, lipo}
set -e
GPU=${1:-0}
DATASET=${2:-bbbp}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."              # repository root, so `python -m magnet...` resolves

# ---- Random split ----
# BBBP optimal: LR=5e-4, ProjDim=256, dropout=0.1, weight-decay=1e-5, fine-tune (gps-lr-ratio=0.1)
echo "===== [random] ${DATASET} ====="
python -m magnet.finetune \
    --split-type random --dataset-name "$DATASET" --gpu "$GPU" \
    --smiles-feature-type concat_all --node-dim 933 \
    --chemberta-model-name DeepChem/ChemBERTa-77M-MTR \
    --pretrained-path pretrain_model/pretrained_gps.pt \
    --split-save-dir data/splits \
    --finetuning-epochs 100 --finetuning-lr 5e-4 \
    --smiles-proj-dim 256 --dropout 0.1 --finetuning-weight-decay 1e-5 \
    --gps-lr-ratio 0.1 \
    --runs 5

# ---- Scaffold split ----
# BBBP optimal: LR=5e-4, ProjDim=512, dropout=0.3, weight-decay=1e-3, fine-tune (gps-lr-ratio=0.5)
echo "===== [scaffold] ${DATASET} ====="
python -m magnet.finetune \
    --split-type scaffold --dataset-name "$DATASET" --gpu "$GPU" \
    --smiles-feature-type concat_all --node-dim 933 \
    --chemberta-model-name DeepChem/ChemBERTa-77M-MTR \
    --pretrained-path pretrain_model/pretrained_gps.pt \
    --split-save-dir data/splits \
    --finetuning-epochs 100 --finetuning-lr 5e-4 \
    --smiles-proj-dim 512 --dropout 0.3 --finetuning-weight-decay 1e-3 \
    --gps-lr-ratio 0.5 \
    --runs 5

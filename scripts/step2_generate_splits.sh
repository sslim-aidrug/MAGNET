#!/bin/bash
# Generate train/val/test split index files (.npz) for all 10 ADMET benchmarks.
# Ratio 0.8/0.1/0.1, random split + balanced scaffold split, over seeds 42-46.
#
# Requires the processed graph pickles ({dataset}_metagraphs.pkl). Set MAGNET_BASE_DIR
# to your project root, or edit SAVE_DIR / DATA_DIR below.
#
# Usage: bash scripts/step2_generate_splits.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."              # repository root, so `python -m magnet...` resolves

BASE_DIR="${MAGNET_BASE_DIR:-$(pwd)}"
SAVE_DIR="${BASE_DIR}/data/splits"
DATA_DIR="${BASE_DIR}/data/metagraphs"
mkdir -p "${SAVE_DIR}"

CLS_DATASETS="bbbp bace sider clintox tox21 hiv toxcast"
REG_DATASETS="esol freesolv lipo"

echo "Save dir: ${SAVE_DIR}"
echo "Data dir: ${DATA_DIR}"

for SEED in 42 43 44 45 46; do
  echo "========== SEED=${SEED} =========="

  # ---- random split (main-table protocol) ----
  for DATASET in ${CLS_DATASETS}; do
    python -m magnet.data_preprocessing.random_split --dataset-name ${DATASET} --seeds ${SEED} \
        --split-type random --task-type classification \
        --split-save-dir ${SAVE_DIR} --graph-pkl ${DATA_DIR}/${DATASET}_metagraphs.pkl
  done
  for DATASET in ${REG_DATASETS}; do
    python -m magnet.data_preprocessing.random_split --dataset-name ${DATASET} --seeds ${SEED} \
        --split-type random --task-type regression \
        --split-save-dir ${SAVE_DIR} --graph-pkl ${DATA_DIR}/${DATASET}_metagraphs.pkl
  done

  # ---- scaffold split, balanced (supplementary protocol) ----
  for DATASET in ${CLS_DATASETS}; do
    python -m magnet.data_preprocessing.scaffold_split --dataset-name ${DATASET} --seeds ${SEED} \
        --split-type scaffold --task-type classification --balanced \
        --split-save-dir ${SAVE_DIR} --graph-pkl ${DATA_DIR}/${DATASET}_metagraphs.pkl
  done
  for DATASET in ${REG_DATASETS}; do
    python -m magnet.data_preprocessing.scaffold_split --dataset-name ${DATASET} --seeds ${SEED} \
        --split-type scaffold --task-type regression --balanced \
        --split-save-dir ${SAVE_DIR} --graph-pkl ${DATA_DIR}/${DATASET}_metagraphs.pkl
  done
done

echo "========== Done (seeds 42-46) =========="

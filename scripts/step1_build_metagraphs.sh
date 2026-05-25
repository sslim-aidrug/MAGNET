#!/bin/bash
# Build meta-graphs from the raw SMILES CSVs (pipeline step 1).
#
# magnet/metagraph/graph_builder.py runs the full pipeline per dataset:
#   fragmentation (BRICS/JT/Murcko) -> meta-graph assembly -> 933D node features
#   (549D RDKit + 384D ChemBERTa-77M-MTR) -> GPS-format graphs with positional encoding.
#
# Output: data/metagraphs/<dataset>_metagraphs.pkl
#
# Usage: bash scripts/step1_build_metagraphs.sh [GPU]
set -e
GPU=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."              # repository root, so `python -m magnet...` resolves

# Pre-training corpus (ZINC250K, ~250K molecules)
python -m magnet.metagraph.graph_builder \
    --dataset-name zinc250k --gpu "$GPU" \
    --csv-path data/raw/zinc250k.csv

# Downstream ADMET benchmarks (10 datasets)
for DS in bbbp bace hiv sider clintox tox21 toxcast esol freesolv lipo; do
    python -m magnet.metagraph.graph_builder \
        --dataset-name "$DS" --gpu "$GPU" \
        --csv-path data/raw/moleculenet/$DS.csv
done

echo "Done. Meta-graphs written to data/metagraphs/"

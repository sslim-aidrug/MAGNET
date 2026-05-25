# MAGNET

**Multi-view Aggregation of Graphs for Neural Embedding of Topologies** — a multi-view molecular graph learning framework for accurate and interpretable ADMET property prediction.

MAGNET represents each molecule as a unified **meta-graph** built from three complementary fragmentations — **BRICS**, **Junction Tree (JT)**, and **Murcko scaffold** — and connects overlapping fragments across views through atom-overlap edges. This enables cross-view message passing between chemically distinct but structurally related substructures. A multi-objective self-supervised pre-training strategy (graph–graph contrastive learning, molecular descriptor regression, and SMILES–graph alignment) yields transferable embeddings that are fine-tuned for downstream property prediction.

This repository accompanies the paper *"Cross-view molecular graph learning enables interpretable ADMET prediction."*

---

## Overview

MAGNET runs in three stages (see Fig. 1 of the paper):

1. **Molecular fragmentation** — each molecule is decomposed with BRICS, JT, and Murcko scaffold.
2. **Meta-graph construction** — fragments from the three views are merged by atom-index identity and connected by atom-overlap edges (cross-view only; self-loops excluded), forming a single tree-like meta-graph.
3. **Pre-training & fine-tuning** — the meta-graph is encoded by a GraphGPS transformer (local MPNN + global multi-head self-attention) and pre-trained with the multi-objective loss

   ```
   L = α · L_GG  +  β · L_P  +  γ · L_SG
   ```

   where `L_GG` is an InfoNCE graph–graph contrastive loss over two node-masked views, `L_P` regresses normalized RDKit descriptors, and `L_SG` is a cross-modal alignment between the graph embedding and a frozen ChemBERTa embedding. The pre-trained encoder is then fine-tuned, combining graph embeddings with ChemBERTa SMILES features.

---

## Repository layout

```
MAGNET/
├── magnet/                              # Python package (run modules with `python -m magnet.<module>`)
│   ├── conf.py                          # Argument / path configuration (--base-dir or $MAGNET_BASE_DIR)
│   ├── gps_model.py                     # GraphGPS encoder (FragmentGPS) + projection heads
│   ├── metagraph/                       # ── Core: meta-graph construction ──
│   │   ├── fragmentation.py             #    BRICS / JT / Murcko fragmentation
│   │   ├── graph_builder.py             #    meta-graph assembly + preprocessing pipeline
│   │   ├── node_features.py             #    node features: 549D RDKit + 384D ChemBERTa = 933D
│   │   └── pos_encoding.py              #    positional / structural encodings
│   ├── pretrain.py                      # Multi-objective pre-training on ZINC250K
│   ├── finetune.py                      # Fine-tuning + multi-seed evaluation on ADMET benchmarks
│   └── data_preprocessing/              # ── Data preprocessing ──
│       ├── random_split.py              #    random 8:1:1 split index generation
│       └── scaffold_split.py            #    balanced scaffold split index generation
├── data/
│   └── raw/                             # Raw input CSVs (tracked); processed graphs are gitignored
│       ├── zinc250k.csv                 #   pre-training corpus (~250K molecules)
│       └── moleculenet/                 #   10 downstream ADMET benchmark CSVs
├── scripts/                            # numbered by pipeline order
│   ├── step1_build_metagraphs.sh        # Build 933D meta-graphs from raw CSVs
│   ├── step2_generate_splits.sh         # Generate all splits (10 datasets, seeds 42-46)
│   ├── step3_pretrain.sh                # Pre-training (multi-objective)
│   └── step4_finetune.sh                # Fine-tuning + multi-seed evaluation
├── requirements.txt
└── README.md
```

The package is organized around four stages — **`metagraph/`** (meta-graph construction, the core), **`pretrain`**, **`finetune`**, and **`data_preprocessing/`** — with `conf.py` and `gps_model.py` as shared components. Run any entry point as a module from the repository root, e.g. `python -m magnet.finetune ...` (the provided scripts do this for you).

> Datasets, processed graph pickles, and model checkpoints are **not** tracked in git (see `.gitignore`). Place them under `data/` and `pretrain_model/` as described below.

---

## Prerequisites

- Python 3.10
- NVIDIA GPU + CUDA (the paper used CUDA 12.8); CPU works but is slow
- PyTorch and PyTorch Geometric matching your CUDA version

## Setup

```bash
git clone https://github.com/sslim-aidrug/MAGNET.git
cd MAGNET

# 1) Create and activate a conda environment (Python 3.10)
conda create -n magnet python=3.10 -y
conda activate magnet

# 2) Install PyTorch for your CUDA toolkit (see https://pytorch.org)
# 3) Install the remaining dependencies
pip install -r requirements.txt
```

All commands below assume the `magnet` environment is active (`conda activate magnet`).

By default the project root is the repository directory. Override it with `--base-dir /path/to/MAGNET` or by exporting `MAGNET_BASE_DIR`.

---

## Data & preprocessing

The **raw input data** (canonical SMILES + labels) is included under `data/raw/`:

- `data/raw/zinc250k.csv` — pre-training corpus (~250K molecules).
- `data/raw/moleculenet/{dataset}.csv` — the 10 downstream ADMET benchmarks (BBBP, BACE, HIV, SIDER, ClinTox, Tox21, ToxCast, ESOL, FreeSolv, Lipo).

These derive from public sources — **MoleculeNet** (https://moleculenet.org/) and the **ZINC250K** subset of ZINC (https://zinc.docking.org/, via the Junction Tree VAE repository https://github.com/wengong-jin/icml18-jtnn). See `data/raw/README.md` for provenance.

The `magnet/metagraph/` package fragments molecules (BRICS/JT/Murcko, in `fragmentation.py`), builds the meta-graphs (`graph_builder.py`), and attaches 933D node features (`node_features.py`: 549D RDKit descriptors + 384D ChemBERTa). The resulting **processed graph pickles are large (multi-GB) and are not tracked in git** — they are regenerated from the raw CSVs above and written to:

```
data/metagraphs/<dataset>_metagraphs.pkl
data/splits/<dataset>/<dataset>-<split>-<seed>.npz
```

Node features are 933D (549D RDKit fragment descriptors + a 384D per-fragment ChemBERTa-77M-MTR embedding).

Build the meta-graphs from the raw CSVs, then generate the train/validation/test splits (8:1:1, seeds 42–46, random and balanced-scaffold):

```bash
bash scripts/step1_build_metagraphs.sh 0    # build 933D meta-graphs (GPU 0)
bash scripts/step2_generate_splits.sh       # generate splits
```

This writes `.npz` index files to `data/splits/`, which the fine-tuning script consumes via `--split-save-dir`.

---

## Pre-training

Pre-train the GraphGPS encoder on the ZINC250K meta-graphs with the multi-objective loss:

```bash
bash scripts/step3_pretrain.sh 0          # GPU 0
```

Key options (`conf.py`): `--pre-epochs`, `--pre-lr`, `--pre-batch-size`, `--node-mask-ratio`, and the loss weights `--fg-weight` (α, L_GG), `--property-weight` (β, L_P), `--smiles-graph-weight` (γ, L_SG). The pre-trained encoder is saved as a `pretrained_gps.pt` checkpoint.

## Fine-tuning

Fine-tune the pre-trained encoder on a downstream benchmark (5 seeds, mean ± std):

```bash
bash scripts/step4_finetune.sh 0 bbbp     # GPU 0, BBBP
```

Useful options: `--split-type {random,scaffold}`, `--dataset-name`, `--finetuning-lr`, `--smiles-proj-dim`, `--dropout`, `--finetuning-weight-decay`, `--pretrained-path`, `--runs`.

## Reproducing the benchmark results

- Datasets are split **8:1:1**. The main paper table uses **random split**; scaffold-split results (5 seeds) are reported in the Supplementary.
- All experiments are repeated over **5 seeds** and reported as mean ± standard deviation.
- The full per-dataset hyperparameters are listed in the Supplementary (Hyperparameters table).

---

## Citation

```bibtex
@article{mo2026magnet,
  title   = {Cross-view molecular graph learning enables interpretable ADMET prediction},
  author  = {Mo, Juhyeon and Lee, Myungjin and Lee, Seungyeon and Lim, Sangsoo},
  journal = {Nature Communications},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Corresponding author: Sangsoo Lim (`sslim@dgu.ac.kr`), Department of Computer Science and Artificial Intelligence, Dongguk University.

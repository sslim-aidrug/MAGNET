# Raw data

Raw molecular data (canonical SMILES + labels) used by MAGNET. These CSVs are the
inputs to graph construction (`magnet/metagraph/graph_builder.py`); the large
processed meta-graph pickles are **not** shipped here and are built from these files.

## Contents

- `zinc250k.csv` — pre-training corpus (~250K molecules).
- `moleculenet/` — the 10 downstream ADMET benchmarks:
  `bbbp, bace, hiv, sider, clintox, tox21, toxcast, esol, freesolv, lipo`.

## Sources

- **MoleculeNet** benchmarks: https://moleculenet.org/ (Wu et al., 2018), distributed via DeepChem.
- **ZINC250K**: the curated 250K subset of ZINC (https://zinc.docking.org/, Sterling & Irwin, 2015),
  as used by the Junction Tree VAE work (https://github.com/wengong-jin/icml18-jtnn, Jin et al., 2018).

The files here are filtered to canonical SMILES with their property labels; please cite the
original sources above when using these datasets.

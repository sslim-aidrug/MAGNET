"""
Meta-graph node features.

Each fragment node is 933D: 549D RDKit fragment descriptors (5 PhysChem + 5 Ring
+ 6 Topological + 7 Pharmacophore + 4 Electronic + 10 Element + 512 Morgan)
concatenated with the 384D ChemBERTa-77M-MTR CLS embedding of the fragment SMILES.
"""

import os
import warnings
from typing import List

import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, GraphDescriptors
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings('ignore')

MORGAN_DIM = 512
FEATURE_DIM = 5 + 5 + 6 + 7 + 4 + 10 + MORGAN_DIM

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
PHARM_FACTORY = AllChem.BuildFeatureFactory(fdefName)
PHARM_TYPES = ['Donor', 'Acceptor', 'Aromatic', 'Hydrophobe',
               'LumpedHydrophobe', 'PosIonizable', 'NegIonizable']


def compute_physchem_features(mol):
    """5D: Physicochemical properties (normalized)"""
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        mw_norm = min(mw / 500.0, 1.0)
        logp_norm = (logp + 5.0) / 15.0
        logp_norm = max(0.0, min(1.0, logp_norm))
        tpsa_norm = min(tpsa / 140.0, 1.0)
        hbd_norm = min(hbd / 5.0, 1.0)
        hba_norm = min(hba / 10.0, 1.0)

        return [mw_norm, logp_norm, tpsa_norm, hbd_norm, hba_norm]
    except:
        return [0.0] * 5


def compute_ring_features(mol):
    """5D: Ring descriptors"""
    try:
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
        num_saturated_rings = rdMolDescriptors.CalcNumSaturatedRings(mol)

        return [
            min(num_rings / 5.0, 1.0),
            min(num_aromatic_rings / 3.0, 1.0),
            min(num_aliphatic_rings / 3.0, 1.0),
            min(num_saturated_rings / 3.0, 1.0),
            1.0 if num_aromatic_rings > 0 else 0.0,
        ]
    except:
        return [0.0] * 5


def compute_topological_features(mol):
    """6D: Topological descriptors"""
    try:
        chi0 = GraphDescriptors.Chi0(mol)
        chi1 = GraphDescriptors.Chi1(mol)

        kappa1 = GraphDescriptors.Kappa1(mol)
        kappa2 = GraphDescriptors.Kappa2(mol)

        balaban_j = GraphDescriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0

        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        density = (2 * n_bonds) / (n_atoms * (n_atoms - 1)) if n_atoms > 1 else 0

        return [
            min(chi0 / 10.0, 1.0),
            min(chi1 / 5.0, 1.0),
            min(kappa1 / 20.0, 1.0),
            min(kappa2 / 10.0, 1.0),
            min(balaban_j / 5.0, 1.0),
            density,
        ]
    except:
        return [0.0] * 6


def compute_pharmacophore_features(mol):
    """7D: Pharmacophore features"""
    try:
        features = PHARM_FACTORY.GetFeaturesForMol(mol)
        counts = {t: 0 for t in PHARM_TYPES}

        for feat in features:
            ftype = feat.GetFamily()
            if ftype in counts:
                counts[ftype] += 1

        n_atoms = mol.GetNumAtoms()
        return [min(counts[t] / max(n_atoms, 1), 1.0) for t in PHARM_TYPES]
    except:
        return [0.0] * 7


def compute_electronic_features(mol):
    """4D: Electronic/Polarity features"""
    try:
        frac_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)

        tpsa = Descriptors.TPSA(mol)
        total_sa = rdMolDescriptors.CalcLabuteASA(mol)
        polar_ratio = tpsa / total_sa if total_sa > 0 else 0

        n_atoms = mol.GetNumAtoms()
        n_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
        hetero_ratio = n_hetero / n_atoms if n_atoms > 0 else 0

        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_bonds = mol.GetNumBonds()
        rot_ratio = n_rot / n_bonds if n_bonds > 0 else 0

        return [frac_sp3, polar_ratio, hetero_ratio, rot_ratio]
    except:
        return [0.0] * 4


def compute_element_features(mol):
    """10D: Element composition features"""
    element_idx = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4,
                   'Cl': 5, 'Br': 6, 'I': 7, 'P': 8, 'B': 9}
    features = [0.0] * 10

    try:
        total_atoms = mol.GetNumAtoms()
        if total_atoms == 0:
            return features

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in element_idx:
                features[element_idx[symbol]] += 1

        features = [f / total_atoms for f in features]
    except:
        pass

    return features


def compute_morgan_features(mol, n_bits=MORGAN_DIM, radius=2):
    """128D: Morgan fingerprint"""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fp)
    except:
        return [0] * n_bits


def compute_enhanced_features(smiles):
    """
    Compute 165D enhanced features for a fragment SMILES
    = 5 physchem + 5 ring + 6 topological + 7 pharmacophore
      + 4 electronic + 10 element + 128 morgan
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * FEATURE_DIM

        physchem = compute_physchem_features(mol)
        ring = compute_ring_features(mol)
        topo = compute_topological_features(mol)
        pharm = compute_pharmacophore_features(mol)
        electronic = compute_electronic_features(mol)
        element = compute_element_features(mol)
        morgan = compute_morgan_features(mol)

        features = physchem + ring + topo + pharm + electronic + element + morgan
        return features
    except:
        return [0.0] * FEATURE_DIM


def GetFragmentFeature(smiles: str) -> List[float]:
    """549D RDKit feature vector for one fragment SMILES (same for BRICS/JT/Murcko)."""
    return compute_enhanced_features(smiles)


def smiles_to_fragment_features(smiles_list, device=None):
    """Compute the 549D RDKit fragment features for a list of SMILES -> tensor [N, 549]."""
    vectors = []
    print(f"Creating fragment feature vectors: {len(smiles_list)} fragments")
    for smiles in smiles_list:
        vectors.append(GetFragmentFeature(smiles))
    result = torch.tensor(np.array(vectors), dtype=torch.float32)
    if device is not None:
        result = result.to(device)
    return result


# --- ChemBERTa-2 per-fragment embedding (384D) ---
CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
CHEMBERTA_DIM = 384
NODE_FEATURE_DIM = FEATURE_DIM + CHEMBERTA_DIM   # 549 + 384 = 933

_chemberta_cache = {}


def _load_chemberta(model_name=CHEMBERTA_MODEL_NAME, device=None):
    """Load and cache the ChemBERTa tokenizer/model on the given device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name not in _chemberta_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        _chemberta_cache[model_name] = (tokenizer, model)
    return _chemberta_cache[model_name]


def chemberta_cls_embeddings(smiles_list, model_name=CHEMBERTA_MODEL_NAME, batch_size=256, device=None):
    """ChemBERTa CLS-token embedding (384D for ChemBERTa-77M-MTR) for each fragment SMILES."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = _load_chemberta(model_name, device)
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(embeddings, dim=0)


def smiles_to_vector(smiles_list, batch_size=256, device=None, output_dim=None):
    """
    Per-fragment node features (933D): 549D RDKit fragment descriptors concatenated
    with the 384D ChemBERTa-77M-MTR CLS embedding of the fragment SMILES.
    """
    rdkit_features = smiles_to_fragment_features(smiles_list, device=None)
    chemberta_features = chemberta_cls_embeddings(smiles_list, batch_size=batch_size, device=device)
    features = torch.cat([rdkit_features, chemberta_features], dim=1).float()
    if device is not None:
        features = features.to(device)
    return features

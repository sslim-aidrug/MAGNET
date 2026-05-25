"""
Molecular fragmentation: BRICS, Junction Tree (JT), and Murcko scaffold.

Each scheme decomposes a molecule into substructure fragments (with atom-index
tracking). The three resulting views are later merged into a meta-graph by
graph_builder.
"""
import re
from collections import defaultdict

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.Scaffolds import MurckoScaffold

def remove_atom_mapping(smiles):
    """Strip atom-map numbers ([X:n] -> [X]) from a SMILES string."""
    return re.sub(r'\:\d+\]', ']', smiles)

def convert_smiles_to_mol_objects(smiles_list):
    """Canonicalize a list of SMILES via RDKit; drop entries that fail to parse."""
    mol_objects = []
    for smiles in smiles_list:
        cleaned_smiles = remove_atom_mapping(smiles)
        try:
            mol = Chem.MolFromSmiles(cleaned_smiles, sanitize=False)
            if mol is not None:
                smiles_with_aromaticity = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                mol_objects.append(smiles_with_aromaticity)
        except Exception as e:
            print(f"Failed to create Mol object from: {cleaned_smiles}, error: {e}")
    return mol_objects

def set_atom_map_numbers(mol):
    """Tag each atom with its index as an atom-map number (for tracking across fragments)."""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def atom_index_data(mol_objects, mol_with_index):
    """Collect (atom index, atom symbol) per molecule."""
    atom_index_data = []
    for i in range(len(mol_objects)):
        atom_data = []
        for atom in mol_with_index[i].GetAtoms():
            atom_index = atom.GetAtomMapNum()
            atom_symbol = atom.GetSymbol()
            atom_data.append({"Atom Index": atom_index, "Atom Symbol": atom_symbol})
        atom_index_data.append(atom_data)

    return atom_index_data


# --- Fragmentation: Murcko scaffold (core ring system + linkers, side chains removed) ---
def murcko_decompose(mol):
    """
    Split a molecule into Murcko scaffold and side-chain fragments by breaking
    bonds that cross the scaffold boundary.

    Returns:
        (atom-index lists per fragment, fragment mols, list of broken bonds)
    """
    murcko_indices = []
    murcko_mols = []
    bonds_to_break_murcko = []

    try:
        set_atom_map_numbers(mol)
    except Exception as e:
        print(f"Error in set_atom_map_numbers: {e}")

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            print("Scaffold is None, returning full molecule as a fragment.")
            indices = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
            murcko_indices.append(indices)
            murcko_mols.append(mol)
            return murcko_indices, murcko_mols, bonds_to_break_murcko
    except Exception as e:
        print(f"Error in MurckoScaffold.GetScaffoldForMol: {e}")

    try:
        scaffold_indices = list(atom.GetAtomMapNum() for atom in scaffold.GetAtoms())
    except Exception as e:
        print(f"Error extracting scaffold indices: {e}")

    try:
        rw_mol = Chem.RWMol(mol)
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            if (atom1.GetAtomMapNum() in scaffold_indices and atom2.GetAtomMapNum() not in scaffold_indices) or \
               (atom2.GetAtomMapNum() in scaffold_indices and atom1.GetAtomMapNum() not in scaffold_indices):
                bonds_to_break_murcko.append([atom1.GetAtomMapNum(), atom2.GetAtomMapNum()])
    except Exception as e:
        print(f"Error in bond analysis: {e}")

    try:
        for atom1_idx, atom2_idx in bonds_to_break_murcko:
            rw_mol.RemoveBond(atom1_idx, atom2_idx)
    except Exception as e:
        print(f"Error removing bonds: {e}")

    try:
        fragment_mols = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=False)
        for frag in fragment_mols:
            murcko_indices.append([atom.GetAtomMapNum() for atom in frag.GetAtoms()])
            murcko_mols.append(frag)
    except Exception as e:
        print(f"Error extracting fragments: {e}")

    return murcko_indices, murcko_mols, bonds_to_break_murcko

def process_murcko_decomposition(data):
    """Run Murcko decomposition over a dataset; collect per-molecule fragments,
    atom indices, broken bonds, and the set of unique fragment SMILES."""
    murcko_error = []
    murcko_indices = []
    murcko_mols = []
    bonds_to_break_murcko = []
    murcko_all_frag = set()

    print(f"Murcko Decomposition: {len(data)} molecules")
    for i in range(len(data)):
        try:
            indices, mols, bonds_to_break = murcko_decompose(Chem.MolFromSmiles(data['smiles'][i]))
            murcko_indices.append(indices)
            bonds_to_break_murcko.append(bonds_to_break)

            smiles_mols = [Chem.MolToSmiles(mol, kekuleSmiles=False) for mol in mols]
            murcko_mols.append(convert_smiles_to_mol_objects(smiles_mols))

            convert_murcko_mols = set(convert_smiles_to_mol_objects(smiles_mols))
            murcko_all_frag = murcko_all_frag.union(convert_murcko_mols)

        except Exception as e:
            murcko_error.append(i)

    return murcko_error, murcko_all_frag, murcko_indices, murcko_mols, bonds_to_break_murcko

# --- Fragmentation: Junction Tree (ring systems and bonds as tree nodes) ---
MST_MAX_WEIGHT = 100
MAX_NCAND = 2000

def tree_decomp(mol):
    """
    Junction-tree decomposition: take ring systems (SSSR) and non-ring bonds as
    cliques, merge heavily-overlapping rings, then build a minimum spanning tree
    over cliques sharing atoms.

    Returns:
        (cliques as atom-index lists, tree edges between cliques)
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2:
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter)

    edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row,col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    return (cliques, edges)

def cliques_to_smiles(mol, cliques):
    """Convert each clique (a set of atom indices) into a fragment SMILES."""
    smiles_list = []
    for clique in cliques:
        atom_indices = list(clique)
        atom_indices.sort()
        emol = Chem.EditableMol(Chem.Mol())

        idx_map = {}
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            new_idx = emol.AddAtom(atom)
            idx_map[idx] = new_idx

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in atom_indices and a2 in atom_indices:
                emol.AddBond(idx_map[a1], idx_map[a2], bond.GetBondType())

        submol = emol.GetMol()
        smiles = Chem.MolToSmiles(submol)
        smiles_list.append(smiles)
    return smiles_list

def process_junction_tree_decomposition(data):
    """Run JT decomposition over a dataset; collect cliques, fragment SMILES,
    tree edges, and the set of unique fragment SMILES."""
    jt_error_indices = []
    jt_all_frag = set()
    jt_indices = []
    jt_mols = []
    bonds_to_break_jt = []

    print(f"JT Decomposition: {len(data)} molecules")
    for i in range(len(data)):
        try:
            mol = Chem.MolFromSmiles(data['smiles'][i])
            cliques, edges = tree_decomp(mol)
            jt_indices.append(cliques)
            bonds_to_break_jt.append(edges)
            mols = cliques_to_smiles(mol, cliques)
            jt_mols.append(mols)
            convert_jt_mols = set(mols)
            jt_all_frag = jt_all_frag.union(convert_jt_mols)
        except Exception as e:
            jt_error_indices.append(i)

    return jt_error_indices, jt_all_frag, jt_indices, jt_mols, bonds_to_break_jt

# --- Fragmentation: BRICS (retrosynthetic bond cleavage) ---
def brics_decompose(mol):
    """
    BRICS decomposition: break retrosynthetically meaningful bonds.

    Returns:
        (atom-index lists per fragment, fragment mols, broken BRICS bonds)
    """
    set_atom_map_numbers(mol)
    rw_mol_brics = RWMol(mol)

    bonds_to_break_brics = list(BRICS.FindBRICSBonds(mol))
    for bond in bonds_to_break_brics:
        atom1, atom2 = bond[0]
        rw_mol_brics.RemoveBond(atom1, atom2)

    brics_fragments = Chem.GetMolFrags(rw_mol_brics, asMols=True, sanitizeFrags=False)
    brics_fragment_indices = []
    brics_mols = []
    for frag in brics_fragments:
        indices = [atom.GetAtomMapNum() for atom in frag.GetAtoms()]
        brics_fragment_indices.append(indices)
        brics_mols.append(frag)

    return brics_fragment_indices, brics_mols, bonds_to_break_brics

def process_brics_decomposition(data):
    """Run BRICS decomposition over a dataset; collect per-molecule fragments,
    atom indices, broken bonds, and the set of unique fragment SMILES."""
    brics_error_indices = []
    brics_all_frag = set()
    brics_indices = []
    brics_mols = []
    bonds_to_break_brics = []

    print(f"BRICS Decomposition: {len(data)} molecules")
    for i in range(len(data)):
        try:
            indices, mols, bonds_to_break = brics_decompose(Chem.MolFromSmiles(data['smiles'][i]))
            brics_indices.append(indices)
            bonds_to_break_brics.append(bonds_to_break)

            mols = [Chem.MolToSmiles(mol) for mol in mols]
            brics_mols.append(convert_smiles_to_mol_objects(mols))

            convert_brics_mols = set(convert_smiles_to_mol_objects(mols))
            brics_all_frag = brics_all_frag.union(convert_brics_mols)

        except Exception as e:
            brics_error_indices.append(i)

    return brics_error_indices, brics_all_frag, brics_indices, brics_mols, bonds_to_break_brics

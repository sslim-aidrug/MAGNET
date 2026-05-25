"""
Meta-graph construction and the full preprocessing pipeline.

Takes the BRICS/JT/Murcko fragment views (from fragmentation.py), merges them into a
unified meta-graph (nodes = fragments merged by atom-index identity; edges connect
overlapping fragments from different views, weighted by the shared-atom fraction),
attaches 933D node features (node_features.py: 549D RDKit + 384D ChemBERTa), and emits
GPS-format graphs with positional encoding (pos_encoding.py).
"""
import os
import copy
import pickle
from itertools import chain, zip_longest
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx

from rdkit import Chem

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

from magnet.conf import get_conf
from magnet.metagraph.node_features import smiles_to_vector
from magnet.metagraph.fragmentation import (
    convert_smiles_to_mol_objects, set_atom_map_numbers, atom_index_data,
    process_murcko_decomposition, process_junction_tree_decomposition,
    process_brics_decomposition,
)



# --- Meta-graph construction: merge fragments across views, build atom-overlap edges ---
def create_atom_fragment_matrix(atom_data, atom_indices, fragments):
    """Build the atom-to-fragment incidence matrix T (rows=atoms, cols=fragments;
    T[i, j] = 1 if atom i belongs to fragment j)."""
    total_atoms = len(atom_data)
    translation_matrix = np.zeros((total_atoms, len(fragments)), dtype=int)

    for frag_idx, indices in enumerate(atom_indices):
        for atom_idx in indices:
            translation_matrix[atom_idx, frag_idx] = 1

    atom_labels = [f"{atom['Atom Symbol']} {atom['Atom Index']}" for atom in atom_data]
    df_translation_matrix = pd.DataFrame(translation_matrix, index=atom_labels, columns=fragments)

    return df_translation_matrix

def frag_dict(brics_translation_matrix, jt_translation_matrix, murcko_translation_matrix):
    """Assign a global node index to every fragment across the three views,
    mapping each fragment SMILES to its index list."""
    matrices = [
        brics_translation_matrix.copy(),
        jt_translation_matrix.copy(),
        murcko_translation_matrix.copy()]

    column_counter = 0

    frag_dict = defaultdict(list)

    for i, matrix in enumerate(matrices):
        columns = matrix.columns.tolist()

        num_columns = matrix.shape[1]
        num_index = range(column_counter, column_counter + num_columns)
        matrix.columns = num_index
        num_index = list(num_index)
        column_counter += num_columns

        result_dict = defaultdict(list)
        for key, value in zip_longest(columns, num_index):
            result_dict[key].append(value)

        for key, value in result_dict.items():
            frag_dict[key].extend(value)

    return dict(frag_dict)

def mk_translation_matrix_with_index(graph_data, brics_translation_matrix, jt_translation_matrix, murcko_translation_matrix):
    """Re-index each per-molecule incidence matrix so columns carry global
    fragment indices that are unique across the three views."""
    brics_translation_matrix_index = copy.deepcopy(brics_translation_matrix)
    jt_translation_matrix_index = copy.deepcopy(jt_translation_matrix)
    murcko_translation_matrix_index = copy.deepcopy(murcko_translation_matrix)

    for data in range(len(graph_data)):
        matrices = [
            brics_translation_matrix_index[data],
            jt_translation_matrix_index[data],
            murcko_translation_matrix_index[data]]

        column_counter = 0
        frag_dict = defaultdict(list)

        for i, matrix in enumerate(matrices):
            columns = matrix.columns.tolist()
            num_columns = matrix.shape[1]
            num_index = range(column_counter, column_counter + num_columns)
            matrix.columns = num_index
            num_index = list(num_index)
            column_counter += num_columns
    return brics_translation_matrix_index, jt_translation_matrix_index, murcko_translation_matrix_index

def calculate_overlap_weights(brics_translation_matrix, jt_translation_matrix, murcko_translation_matrix):
    """Compute cross-view edge weights from atom overlap (normalized by fragment
    size) for each view pair: BRICS-JT, JT-Murcko, BRICS-Murcko."""
    def compute_adjacency_matrix(matrix_a, matrix_b):
        indices_a = matrix_a.columns.tolist()
        indices_b = matrix_b.columns.tolist()
        indices = indices_a + indices_b

        # Atom-overlap between fragments of two views: O = T_a^T T_b (shared-atom counts)
        overlap_matrix = np.dot(matrix_a.T, matrix_b)
        overlap_df = pd.DataFrame(overlap_matrix, index=indices_a, columns=indices_b)

        fragment_atom_a = matrix_a.sum().to_dict()
        fragment_atom_b = matrix_b.sum().to_dict()

        num_a = len(indices_a)
        num_b = len(indices_b)
        total_nodes = num_a + num_b

        adjacency_matrix = np.zeros((total_nodes, total_nodes))
        adjacency_df = pd.DataFrame(adjacency_matrix, index=indices, columns=indices)

        frag_atoms_a_arr = np.array([fragment_atom_a.get(i, 1) for i in indices_a])
        frag_atoms_b_arr = np.array([fragment_atom_b.get(i, 1) for i in indices_b])

        # Edge weight = shared atoms normalized by fragment size (directional, both ways)
        normalized_a_to_b = overlap_df.values / frag_atoms_a_arr[:, None]
        normalized_b_to_a = overlap_df.values.T / frag_atoms_b_arr[:, None]

        adjacency_df.iloc[0:num_a, num_a:total_nodes] = np.round(normalized_a_to_b, 3)
        adjacency_df.iloc[num_a:total_nodes, 0:num_a] = np.round(normalized_b_to_a, 3)

        return adjacency_df

    # Edges only between different views (BRICS-JT, JT-Murcko, BRICS-Murcko); never within a view
    brics_jt = compute_adjacency_matrix(brics_translation_matrix, jt_translation_matrix)
    jt_murcko = compute_adjacency_matrix(jt_translation_matrix, murcko_translation_matrix)
    brics_murcko = compute_adjacency_matrix(brics_translation_matrix, murcko_translation_matrix)

    return [brics_jt, jt_murcko, brics_murcko]


def combine_overlap_weights(all_overlap_weights_data):
    """Merge the three pairwise overlap-weight matrices into one combined
    node x node weight matrix over all fragments."""
    brics_jt, jt_murcko, brics_murcko = all_overlap_weights_data

    combined_indices = sorted(set(brics_jt.index).union(set(jt_murcko.index), set(brics_murcko.index)))

    total_nodes = len(combined_indices)
    combined_df = pd.DataFrame(np.zeros((total_nodes, total_nodes)),
                               index=combined_indices, columns=combined_indices)

    row_indices = [combined_indices.index(i) for i in brics_jt.index]
    col_indices = [combined_indices.index(i) for i in brics_jt.columns]
    combined_df.values[np.ix_(row_indices, col_indices)] = brics_jt.values

    row_indices = [combined_indices.index(i) for i in jt_murcko.index]
    col_indices = [combined_indices.index(i) for i in jt_murcko.columns]
    combined_df.values[np.ix_(row_indices, col_indices)] = jt_murcko.values

    row_indices = [combined_indices.index(i) for i in brics_murcko.index]
    col_indices = [combined_indices.index(i) for i in brics_murcko.columns]
    combined_df.values[np.ix_(row_indices, col_indices)] = brics_murcko.values

    return combined_df

def build_frag_dict_from_translation_matrices(
    brics_frag_all, jt_frag_all, murcko_frag_all,
    brics_idx_all, jt_idx_all, murcko_idx_all):
    """Per molecule, build a dict keyed by atom-index set ->
    [fragment SMILES, source-view tags]. Fragments with identical atom sets
    across views are merged (their view tags are appended together)."""
    all_frag_dicts = []

    for data in range(len(brics_frag_all)):
        brics_frag = brics_frag_all[data].columns.tolist()
        brics_idx = brics_idx_all[data].columns.tolist()
        tagged_brics = [f"brics_{i}" for i in brics_idx]
        brics_atom_idx = brics_idx_all[data].values.T.tolist()

        frag_dict = {
            tuple(brics_atom_idx[i]): [brics_frag[i], [tagged_brics[i]]]
            for i in range(len(brics_frag))}

        jt_frag = jt_frag_all[data].columns.tolist()
        jt_idx = jt_idx_all[data].columns.tolist()
        tagged_jt = [f"jt_{i}" for i in jt_idx]
        jt_atom_idx = jt_idx_all[data].values.T.tolist()

        for i in range(len(jt_frag)):
            atom_idx_tuple = tuple(jt_atom_idx[i])
            tag = tagged_jt[i]
            if atom_idx_tuple in frag_dict:
                frag_dict[atom_idx_tuple][1].append(tag)
            else:
                frag_dict[atom_idx_tuple] = [jt_frag[i], [tag]]

        murcko_frag = murcko_frag_all[data].columns.tolist()
        murcko_idx = murcko_idx_all[data].columns.tolist()
        tagged_murcko = [f"murcko_{i}" for i in murcko_idx]
        murcko_atom_idx = murcko_idx_all[data].values.T.tolist()

        for i in range(len(murcko_frag)):
            atom_idx_tuple = tuple(murcko_atom_idx[i])
            tag = tagged_murcko[i]
            if atom_idx_tuple in frag_dict:
                frag_dict[atom_idx_tuple][1].append(tag)
            else:
                frag_dict[atom_idx_tuple] = [murcko_frag[i], [tag]]

        all_frag_dicts.append(frag_dict)

    return all_frag_dicts

def merge_node_dictionary(frag_dict):
    """Assign final node indices: single-view fragments keep their index;
    fragments shared across views (merged) receive a new index."""
    node_indices = [
        int(tag.split('_')[-1])
        for _, (_, tagged_list) in frag_dict.items()
        for tag in tagged_list
    ]
    M = max(node_indices)

    new_dict = {}
    next_new_idx = M + 1

    for atom_idx_tuple, (smiles, tagged_list) in frag_dict.items():
        node_idx = [int(tag.split('_')[-1]) for tag in tagged_list]

        if len(tagged_list) == 1:
            key = node_idx[0]
        else:
            key = next_new_idx
            next_new_idx += 1

        new_dict[key] = [smiles, tagged_list]

    sorted_dict = dict(sorted(new_dict.items()))
    return sorted_dict

def merge_weights(combine_all_overlap_weights, dict_list, verbose=False):
    """Rebuild the weight matrix after node merging: a merged node takes the
    averaged incoming weights and the shared outgoing weights of its source fragments."""
    tag_idx = list(dict_list.keys())
    original_idx = list(combine_all_overlap_weights.keys())

    new_nodes = [x for x in tag_idx if x not in original_idx]
    drop_nodes = [x for x in original_idx if x not in tag_idx]

    tagging_number_groups = [
        [int(tag.split('_')[-1]) for tag in tag_list]
        for _, tag_list in dict_list.values()
        if len(tag_list) >= 2
    ]

    filtered_matrix = combine_all_overlap_weights.drop(index=drop_nodes, columns=drop_nodes)
    combined_df = filtered_matrix.reindex(index=tag_idx, columns=tag_idx, fill_value=0)

    for group, new_col in zip(tagging_number_groups, new_nodes):
        row_avg = round(combine_all_overlap_weights.loc[group].mean(axis=0), 3)
        common_cols = combined_df.columns.intersection(row_avg.index)
        combined_df.loc[new_col, common_cols] = row_avg[common_cols].values

        reference_col = group[0]
        col_vals = combine_all_overlap_weights[reference_col]
        valid_rows = combined_df.index.intersection(col_vals.index)
        combined_df.loc[valid_rows, new_col] = col_vals[valid_rows]

        if verbose:
            print(f"[{new_col}] <- from group {group}")
            print("Row mean inserted:\n", row_avg[common_cols])
            print("Column inserted:\n", col_vals[valid_rows])
            print("-" * 50)
    return combined_df


def change_to_adjacency_matrix_of_combine_overlap_weight(combine_overlap_weight):
    """Binarize the weight matrix into an adjacency matrix (nonzero -> 1)."""
    adjacency_matrix = combine_overlap_weight.copy()
    adjacency_matrix.values[adjacency_matrix.values != 0] = 1
    return adjacency_matrix

def Create_Final_Graph(tagged_smiles_dict, weight_df):
    """Build the meta-graph (NetworkX DiGraph with symmetric edges) from the merged
    node dict and the combined weight matrix; one node per fragment, weighted edges."""
    G = nx.DiGraph()

    G.add_nodes_from(tagged_smiles_dict.keys())

    for row in weight_df.index:
        for col in weight_df.columns:
            wt = weight_df.loc[row, col]
            if wt != 0:
                G.add_edge(row, col, weight=wt)
                G.add_edge(col, row, weight=wt) 
    return G


def find_key_by_value(dictionary, target):
    """Return the fragment SMILES stored for a given node key."""
    return [values[0] for key, values in dictionary.items() if target == key]


from magnet.metagraph.pos_encoding import compute_posenc_stats

# --- Graph assembly (GPS format) and full preprocessing pipeline ---
def create_graphs_with_adjacency(graph_data, frag_dict_list, final_graph, vector,
                                  combine_all_overlap_weights, adjacency_matrix_for_combine_all_overlap_weights,
                                  pe_types=None, max_freqs=8, eigvec_norm="L2"):
    """
    Build graphs in GPS format (with Laplacian PE and degree encoding).

    Output Data format:
        x: [N, 933] - node features (549D RDKit descriptors: 5 physchem + 5 ring + 6 topo + 7 pharm + 4 electronic + 10 element + 512 Morgan; + 384D ChemBERTa-77M-MTR CLS)
        edge_index: [2, E] - sparse edge connections
        edge_attr: [E, 1] - edge weights (overlap)
        pe: [N, max_freqs] - Laplacian positional encoding
        eigvals_pe: [max_freqs] - eigenvalues
        in_degree: [N] - in-degree for centrality
        out_degree: [N] - out-degree for centrality

    Args:
        pe_types: list of PE types (default: ['LapPE', 'Degree'])
        max_freqs: LapPE dimension (default: 8)
        eigvec_norm: eigenvector normalization scheme ('L2', 'L1', 'abs-max', 'wavelength', ...)
    """
    if pe_types is None:
        pe_types = ['LapPE', 'Degree']

    graphs = []

    print(f"Creating GPS format graphs: {len(graph_data)} graphs")
    for data in range(len(graph_data)):
        key_list = list(chain.from_iterable(find_key_by_value(frag_dict_list[data], i) for i in list(np.sort(final_graph[data].nodes))))
        node_features = torch.stack([vector[key] for key in key_list])
        num_nodes = node_features.size(0)

        edge_weight_dense = torch.tensor(combine_all_overlap_weights[data].values, dtype=torch.float)

        adjacency_dense = torch.tensor(adjacency_matrix_for_combine_all_overlap_weights[data].values, dtype=torch.float)

        edge_index, _ = dense_to_sparse(adjacency_dense)

        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            edge_attr = edge_weight_dense[src, dst].unsqueeze(-1).float()
        else:
            edge_attr = torch.zeros(0, 1)

        graph = Data(
            x=node_features.float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        graph = compute_posenc_stats(
            graph,
            pe_types=pe_types,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm
        )

        graphs.append(graph)

    return graphs


def preprocessing(df, dataset_name=''):
    """
    Preprocessing pipeline for meta-graph construction.

    Args:
        df: DataFrame with 'smiles' column
        dataset_name: Name of the dataset
    """
    # 1) Parse molecules and tag each atom with its index
    mol_objects = [Chem.MolFromSmiles(mol, sanitize=True) for mol in convert_smiles_to_mol_objects(df['smiles'])]
    mol_with_index = [set_atom_map_numbers(mol) for mol in copy.deepcopy(mol_objects)]
    all_atom_index_data = atom_index_data(mol_objects, mol_with_index)
    
    # 2) Decompose each molecule with the three views (Murcko, JT, BRICS)
    murcko_error, murcko_all_frag, murcko_indices, murcko_mols, bonds_to_break_murcko = process_murcko_decomposition(df)
    jt_error, jt_all_frag, jt_indices, jt_mols, bonds_to_break_jt = process_junction_tree_decomposition(df)
    brics_error, brics_all_frag, brics_indices, brics_mols, bonds_to_break_brics = process_brics_decomposition(df)
    
    # Drop molecules where any decomposition failed
    error_indices = list(set(murcko_error + jt_error + brics_error))
    filtered_data = df.iloc[[i for i in range(len(df)) if i not in error_indices]].reset_index(drop=True)
    print(f"[{dataset_name}] error indices: BRICS={len(brics_error)}, JT={len(jt_error)}, Murcko={len(murcko_error)}")

    jt_translation_matrix_all = []; brics_translation_matrix_all = []; murcko_translation_matrix_all = []

    # 3) Per-molecule atom-to-fragment incidence matrices, one per view
    for data in range(len(filtered_data)):
        jt_translation_matrix_all.append(create_atom_fragment_matrix(all_atom_index_data[data], jt_indices[data], jt_mols[data]))
        brics_translation_matrix_all.append(create_atom_fragment_matrix(all_atom_index_data[data], brics_indices[data], brics_mols[data]))
        murcko_translation_matrix_all.append(create_atom_fragment_matrix(all_atom_index_data[data], murcko_indices[data], murcko_mols[data]))

    brics_translation_matrix_index_all, jt_translation_matrix_index_all, murcko_translation_matrix_index_all = mk_translation_matrix_with_index(filtered_data, brics_translation_matrix_all, jt_translation_matrix_all, murcko_translation_matrix_all)
    print(f"Creating all overlap weights: {len(filtered_data)} molecules")
    
    # 4) Cross-view atom-overlap edge weights, then combine the three view pairs
    all_overlap_weights = [calculate_overlap_weights(brics_translation_matrix_index_all[data], jt_translation_matrix_index_all[data], murcko_translation_matrix_index_all[data]) for data in range(len(filtered_data))]
    combine_all_overlap_weights = [combine_overlap_weights(all_overlap_weights[data]) for data in range(len(filtered_data))]
    
    # 5) Group fragments by atom-index set across the three views
    all_frag_dict_list = build_frag_dict_from_translation_matrices(
                                    brics_translation_matrix_all, jt_translation_matrix_all, murcko_translation_matrix_all,
                                    brics_translation_matrix_index_all, jt_translation_matrix_index_all, murcko_translation_matrix_index_all)

    index_per_method = {}
    for i in range(len(brics_translation_matrix_index_all)):
        index_per_method[f'{dataset_name}_{i}'] = {
            "brics":list(brics_translation_matrix_index_all[i].columns),
            "jt" : list(jt_translation_matrix_index_all[i].columns),
            "murcko" : list(murcko_translation_matrix_index_all[i].columns)}


    # 6) Merge fragments shared across views into single nodes and rebuild weights
    tagged_smiles_dict_list = [merge_node_dictionary(frag_dict) for frag_dict in all_frag_dict_list]
    print(f"Creating merge combine all overlap weights: {len(filtered_data)} molecules")
    merge_combine_all_overlap_weights = [merge_weights(combine_all_overlap_weights[data], tagged_smiles_dict_list[data]) for data in range(len(filtered_data))]
    adjacency_matrix_for_combine_all_overlap_weights = [change_to_adjacency_matrix_of_combine_overlap_weight(merge_combine_all_overlap_weights[data]) for data in range(len(filtered_data))]
    print(f"Creating Final Graph: {len(filtered_data)} molecules")
    
    # 7) Assemble the meta-graph per molecule
    final_graphs = [Create_Final_Graph(tagged_smiles_dict_list[data], merge_combine_all_overlap_weights[data]) for data in range(len(filtered_data))]

    # 8) Compute 549D node features for every unique fragment
    all_frag_dataset = list(brics_all_frag|murcko_all_frag|jt_all_frag)
    vectors = smiles_to_vector(all_frag_dataset)
    smiles_to_vectors = {smiles: vec for smiles, vec in zip(all_frag_dataset, vectors)}

    # 9) Convert to GPS-format graphs (node features + positional encoding + edges)
    all_graphs = create_graphs_with_adjacency(filtered_data, tagged_smiles_dict_list, final_graphs, smiles_to_vectors, merge_combine_all_overlap_weights, adjacency_matrix_for_combine_all_overlap_weights)

    return filtered_data, mol_objects, mol_with_index, all_atom_index_data, brics_mols, jt_mols, murcko_mols, filtered_data, final_graphs, all_graphs, all_frag_dict_list, tagged_smiles_dict_list, index_per_method


if __name__ == '__main__':
    args = get_conf()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    csv_path = getattr(args, "csv_path", None)
    ds_name = args.dataset_name.strip().lower()

    if csv_path is None or csv_path == "":
        base_dir = Path(args.base_dir)
        data_dir = base_dir / "data/raw"
        print(f"[graph_builder] dataset_name: {ds_name}")
        csv_path = data_dir / f"{ds_name}.csv"
    else:
        csv_path = Path(csv_path)

    print(f"[graph_builder] csv_path: {csv_path}")

    df = pd.read_csv(csv_path)
    if ds_name in ["bbbp", "bace", "clintox", "sider", "tox21"]:
        before = len(df)
        df = df[df['smiles'].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)]
        after = len(df)
        print(f"[graph_builder] {ds_name}: removed {before - after} invalid SMILES ({after} remaining)")

    result = preprocessing(df, dataset_name=args.dataset_name)
    prefix = args.dataset_name

    data_bundle = {
        f"{prefix}_filtered_data": result[0],
        f"{prefix}_mol_objects": result[1],
        f"{prefix}_mol_with_index": result[2],
        f"{prefix}_all_atom_index_data": result[3],
        f"{prefix}_brics_mols": result[4],
        f"{prefix}_jt_mols": result[5],
        f"{prefix}_murcko_mols": result[6],
        f"{prefix}_final_graphs": result[8],
        f"{prefix}_all_graphs": result[9],
        f"{prefix}_all_frag_dict_list": result[10],
        f"{prefix}_tagged_smiles_dict_list": result[11],
        f"{prefix}_index_per_method": result[12],}

    os.makedirs(os.path.dirname(args.graph_pkl), exist_ok=True)

    with open(args.graph_pkl, 'wb') as f:
        pickle.dump(data_bundle, f)

    print(f'Saved graph data: {args.graph_pkl}')
    print(f'{prefix} filtered_data rows : {len(data_bundle[f"{prefix}_filtered_data"])}')
    print(f'{prefix} all_graphs count   : {len(data_bundle[f"{prefix}_all_graphs"])}')

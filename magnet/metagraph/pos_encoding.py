"""
Positional encoding for meta-graphs.

Reference: https://github.com/rampasek/GraphGPS
- Laplacian Positional Encoding (LapPE)
- Random Walk Structural Encoding (RWSE)
"""

import torch
import numpy as np
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
from torch_scatter import scatter


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Eigenvector normalization.

    Args:
        EigVecs: eigenvectors [N, k]
        EigVals: eigenvalues [k]
        normalization: scheme ('L1', 'L2', 'abs-max', 'wavelength', 'wavelength-asin', 'wavelength-soft')
        eps: small value for numerical stability

    Returns:
        normalized eigenvectors
    """
    EigVals = EigVals.unsqueeze(0) if EigVals.dim() == 1 else EigVals

    if normalization == "L1":
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom

    elif normalization == "wavelength-asin":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom
        EigVecs = torch.asin(EigVecs / (denom + eps))
        return EigVecs

    elif normalization == "wavelength-soft":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        denom = denom * (eigval_denom + 1)

    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    denom[denom < eps] = 1
    return EigVecs / denom


def compute_laplacian_pe(data, max_freqs=8, eigvec_norm="L2"):
    """
    Compute Laplacian Positional Encoding.

    Args:
        data: PyG Data object (requires edge_index)
        max_freqs: number of eigenvectors to use (pe_dim)
        eigvec_norm: eigenvector normalization scheme

    Returns:
        data: Data object with pe, eigvals_pe added
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if edge_index.numel() == 0:
        data.pe = torch.zeros(num_nodes, max_freqs)
        data.eigvals_pe = torch.zeros(max_freqs)
        return data

    edge_index_lap, edge_weight = get_laplacian(
        edge_index,
        normalization='sym',
        num_nodes=num_nodes
    )

    L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)

    try:
        EigVals, EigVecs = np.linalg.eigh(L.toarray())
    except Exception as e:
        data.pe = torch.zeros(num_nodes, max_freqs)
        data.eigvals_pe = torch.zeros(max_freqs)
        return data

    EigVals = torch.from_numpy(EigVals).float()
    EigVecs = torch.from_numpy(EigVecs).float()

    idx = EigVals.argsort()
    EigVals = EigVals[idx]
    EigVecs = EigVecs[:, idx]

    k = min(max_freqs, num_nodes)
    EigVals = EigVals[:k]
    EigVecs = EigVecs[:, :k]

    EigVals = EigVals.clamp_min(0)

    EigVecs = eigvec_normalizer(EigVecs, EigVals, normalization=eigvec_norm)

    if k < max_freqs:
        pad_vecs = torch.zeros(num_nodes, max_freqs - k)
        pad_vals = torch.zeros(max_freqs - k)
        EigVecs = torch.cat([EigVecs, pad_vecs], dim=1)
        EigVals = torch.cat([EigVals, pad_vals], dim=0)


    data.pe = EigVecs
    data.eigvals_pe = EigVals

    return data


def compute_rwse(data, ksteps=None):
    """
    Compute Random Walk Structural Encoding.

    Args:
        data: PyG Data object
        ksteps: list of steps to compute (default: [1,2,3,...,16])

    Returns:
        data: Data object with rwse added
    """
    if ksteps is None:
        ksteps = list(range(1, 17))

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if edge_index.numel() == 0:
        data.rwse = torch.zeros(num_nodes, len(ksteps))
        return data

    edge_weight = data.edge_attr.squeeze(-1) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    source, dest = edge_index[0], edge_index[1]

    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = 1.0 / deg.clamp(min=1e-12)

    edge_weight_norm = edge_weight * deg_inv[source]

    P = to_dense_adj(edge_index, edge_attr=edge_weight_norm, max_num_nodes=num_nodes).squeeze(0)

    rw_landing = []
    Pk = P.clone()

    for k in range(1, max(ksteps) + 1):
        if k > 1:
            Pk = Pk @ P
        if k in ksteps:
            rw_landing.append(Pk.diagonal().unsqueeze(1))

    data.rwse = torch.cat(rw_landing, dim=1)

    return data


def compute_degree_encoding(data, max_degree=100):
    """
    Compute degree (centrality) encoding.

    Args:
        data: PyG Data object
        max_degree: maximum degree value (for clipping)

    Returns:
        data: Data object with in_degree, out_degree added
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if edge_index.numel() == 0:
        data.in_degree = torch.zeros(num_nodes, dtype=torch.long)
        data.out_degree = torch.zeros(num_nodes, dtype=torch.long)
        return data

    row, col = edge_index

    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    in_degree.scatter_add_(0, col, torch.ones_like(col, dtype=torch.long))

    out_degree = torch.zeros(num_nodes, dtype=torch.long)
    out_degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.long))

    data.in_degree = in_degree.clamp(max=max_degree)
    data.out_degree = out_degree.clamp(max=max_degree)

    return data


def compute_posenc_stats(data, pe_types=None, max_freqs=8, eigvec_norm="L2",
                          rw_ksteps=None, max_degree=100):
    """
    Compute positional encodings.

    Args:
        data: PyG Data object
        pe_types: list of PE types to compute ['LapPE', 'RWSE', 'Degree']
        max_freqs: LapPE dimension
        eigvec_norm: eigenvector normalization scheme
        rw_ksteps: RWSE step list
        max_degree: maximum degree

    Returns:
        data: Data object with PE added
    """
    if pe_types is None:
        pe_types = ['LapPE', 'Degree']

    if 'LapPE' in pe_types:
        data = compute_laplacian_pe(data, max_freqs=max_freqs, eigvec_norm=eigvec_norm)

    if 'RWSE' in pe_types:
        data = compute_rwse(data, ksteps=rw_ksteps)

    if 'Degree' in pe_types:
        data = compute_degree_encoding(data, max_degree=max_degree)

    return data


def add_posenc_to_graphs(graphs, pe_types=None, max_freqs=8, eigvec_norm="L2",
                          rw_ksteps=None, max_degree=100, verbose=True):
    """
    Add positional encodings to a list of graphs.

    Args:
        graphs: list of PyG Data objects
        pe_types: PE types
        max_freqs: LapPE dimension
        eigvec_norm: normalization scheme
        rw_ksteps: RWSE steps
        max_degree: maximum degree
        verbose: print progress

    Returns:
        list of graphs with PE added
    """
    from tqdm import tqdm

    if pe_types is None:
        pe_types = ['LapPE', 'Degree']

    iterator = tqdm(graphs, desc="Computing PE stats") if verbose else graphs

    result = []
    for data in iterator:
        try:
            data = compute_posenc_stats(
                data,
                pe_types=pe_types,
                max_freqs=max_freqs,
                eigvec_norm=eigvec_norm,
                rw_ksteps=rw_ksteps,
                max_degree=max_degree
            )
            result.append(data)
        except Exception as e:
            num_nodes = data.num_nodes
            if 'LapPE' in pe_types:
                data.pe = torch.zeros(num_nodes, max_freqs)
                data.eigvals_pe = torch.zeros(max_freqs)
            if 'RWSE' in pe_types:
                ksteps = rw_ksteps if rw_ksteps else list(range(1, 17))
                data.rwse = torch.zeros(num_nodes, len(ksteps))
            if 'Degree' in pe_types:
                data.in_degree = torch.zeros(num_nodes, dtype=torch.long)
                data.out_degree = torch.zeros(num_nodes, dtype=torch.long)
            result.append(data)

    return result

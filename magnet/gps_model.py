"""
GraphGPS Model Implementation for Fragment-level Molecular Graphs

Reference: https://github.com/rampasek/GraphGPS
- GPS Layer: Message Passing + Transformer (MPNN + Global Attention)
- Positional/Structural Encoding: LapPE, RWSE, Degree Encoding

Data format (from convert_to_gps.py):
    data.x: [N, 768] - ChemBERTa fragment embeddings
    data.edge_index: [2, E] - sparse edge connections
    data.edge_attr: [E, 1] - edge weights (overlap degree)
    data.pe: [N, max_freqs] - Laplacian positional encoding
    data.eigvals_pe: [max_freqs] - eigenvalues
    data.in_degree: [N] - in-degree
    data.out_degree: [N] - out-degree
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GINEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class DegreeEncoder(nn.Module):
    """
    Degree Encoding (Centrality Encoding from Graphormer)
    Embeds in-degree and out-degree as learnable vectors
    """
    def __init__(self, emb_dim, max_degree=100):
        super().__init__()
        self.max_degree = max_degree
        self.in_degree_encoder = nn.Embedding(max_degree + 1, emb_dim)
        self.out_degree_encoder = nn.Embedding(max_degree + 1, emb_dim)

    def forward(self, in_degree, out_degree):
        """
        Args:
            in_degree: [N] - in-degree values
            out_degree: [N] - out-degree values
        Returns:
            degree_emb: [N, emb_dim]
        """
        in_degree = in_degree.clamp(0, self.max_degree)
        out_degree = out_degree.clamp(0, self.max_degree)
        return self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)


class LapPEEncoder(nn.Module):
    """
    Laplacian Positional Encoding Encoder
    Projects eigenvectors to embedding dimension with optional sign-flip augmentation
    """
    def __init__(self, pe_dim, emb_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = pe_dim if i == 0 else emb_dim
            layers.append(nn.Linear(in_dim, emb_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, pe):
        """
        Args:
            pe: [N, pe_dim] - Laplacian eigenvectors
        Returns:
            pe_emb: [N, emb_dim]
        """
        return self.encoder(pe)


class RWSEEncoder(nn.Module):
    """
    Random Walk Structural Encoding Encoder
    """
    def __init__(self, rw_dim, emb_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = rw_dim if i == 0 else emb_dim
            layers.append(nn.Linear(in_dim, emb_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, rwse):
        """
        Args:
            rwse: [N, rw_dim] - Random walk structural encoding
        Returns:
            rw_emb: [N, emb_dim]
        """
        return self.encoder(rwse)


class EdgeEncoder(nn.Module):
    """
    Edge Encoder for continuous edge weights
    Projects edge attributes to embedding dimension
    """
    def __init__(self, edge_dim, emb_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(edge_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: [E, edge_dim] - edge attributes
        Returns:
            edge_emb: [E, emb_dim]
        """
        return self.encoder(edge_attr)


class GPSLayer(nn.Module):
    """
    GPS Layer: MPNN + Global Attention

    From the paper "Recipe for a General, Powerful, Scalable Graph Transformer"
    Combines local message passing with global attention mechanism
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1,
                 attn_dropout=0.1, local_gnn_type='GINE', global_model_type='Transformer'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type

        if local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.local_model = GINEConv(gin_nn, edge_dim=hidden_dim)
        elif local_gnn_type == 'GATv2':
            self.local_model = GATv2Conv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, edge_dim=hidden_dim, concat=True
            )
        else:
            self.local_model = None

        if global_model_type == 'Transformer':
            self.self_attn = nn.MultiheadAttention(
                hidden_dim, num_heads,
                dropout=attn_dropout, batch_first=True
            )
        else:
            self.self_attn = None

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: [N, hidden_dim] - node features
            edge_index: [2, E] - edge connections
            edge_attr: [E, hidden_dim] - edge embeddings
            batch: [N] - batch assignment for nodes
        Returns:
            x: [N, hidden_dim] - updated node features
        """
        # Local message passing over meta-graph edges (residual)
        if self.local_model is not None:
            h = self.norm1(x)
            h_local = self.local_model(h, edge_index, edge_attr)
            x = x + self.dropout(h_local)

        # Global self-attention across all fragment nodes (residual)
        if self.self_attn is not None:
            h = self.norm2(x)

            padded, mask = to_dense_batch(h, batch)

            key_padding_mask = ~mask

            h_global, _ = self.self_attn(padded, padded, padded, key_padding_mask=key_padding_mask)

            h_out = h_global[mask]

            x = x + self.dropout(h_out)

        h = self.norm3(x)
        x = x + self.ffn(h)

        return x


class FragmentGPS(nn.Module):
    """
    GraphGPS model for Fragment-level Molecular Property Prediction

    Architecture:
        1. Node Feature Projection (768 -> hidden_dim)
        2. Positional/Structural Encoding (LapPE, Degree, optional RWSE)
        3. Edge Encoding
        4. Stacked GPS Layers (MPNN + Transformer)
        5. Graph-level Readout (mean/sum pooling)
        6. Output Head (for prediction)
    """
    def __init__(
        self,
        node_dim=768,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        attn_dropout=0.1,
        pe_dim=8,
        use_lap_pe=True,
        use_degree=True,
        use_rwse=False,
        rwse_dim=16,
        edge_dim=1,
        output_dim=None,
        pool='mean',
        local_gnn_type='GINE',
        global_model_type='Transformer',
        max_degree=100
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_lap_pe = use_lap_pe
        self.use_degree = use_degree
        self.use_rwse = use_rwse
        self.pool = pool

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if use_lap_pe:
            self.lap_encoder = LapPEEncoder(pe_dim, hidden_dim)

        if use_degree:
            self.degree_encoder = DegreeEncoder(hidden_dim, max_degree)

        if use_rwse:
            self.rwse_encoder = RWSEEncoder(rwse_dim, hidden_dim)

        self.edge_encoder = EdgeEncoder(edge_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GPSLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        if output_dim is not None:
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.output_head = None

    def forward(self, data):
        """
        Args:
            data: PyG Batch object containing:
                - x: [N, node_dim]
                - edge_index: [2, E]
                - edge_attr: [E, edge_dim]
                - pe: [N, pe_dim] (optional)
                - in_degree: [N] (optional)
                - out_degree: [N] (optional)
                - rwse: [N, rwse_dim] (optional)
                - batch: [N]

        Returns:
            If output_head: [batch_size, output_dim]
            Else: [N, hidden_dim] (node embeddings) or [batch_size, hidden_dim] (graph embeddings)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        h = self.node_encoder(x.float())

        if self.use_lap_pe and hasattr(data, 'pe') and data.pe is not None:
            pe = data.pe.float()
            if self.training:
                sign_flip = torch.randint(0, 2, (pe.size(1),), device=pe.device) * 2 - 1
                pe = pe * sign_flip.unsqueeze(0)
            h = h + self.lap_encoder(pe)

        if self.use_degree:
            if hasattr(data, 'in_degree') and hasattr(data, 'out_degree'):
                h = h + self.degree_encoder(data.in_degree, data.out_degree)

        if self.use_rwse and hasattr(data, 'rwse') and data.rwse is not None:
            h = h + self.rwse_encoder(data.rwse.float())

        if edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr.float())
        else:
            edge_emb = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)

        for layer in self.layers:
            h = layer(h, edge_index, edge_emb, batch)

        h = self.final_norm(h)

        if self.pool == 'mean':
            graph_emb = global_mean_pool(h, batch)
        elif self.pool == 'sum':
            graph_emb = global_add_pool(h, batch)
        elif self.pool == 'max':
            graph_emb = global_max_pool(h, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pool}")

        if self.output_head is not None:
            return self.output_head(graph_emb)

        return graph_emb

    def get_node_embeddings(self, data):
        """
        Get node-level embeddings without pooling
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        h = self.node_encoder(x.float())

        if self.use_lap_pe and hasattr(data, 'pe') and data.pe is not None:
            h = h + self.lap_encoder(data.pe.float())

        if self.use_degree:
            if hasattr(data, 'in_degree') and hasattr(data, 'out_degree'):
                h = h + self.degree_encoder(data.in_degree, data.out_degree)

        if self.use_rwse and hasattr(data, 'rwse') and data.rwse is not None:
            h = h + self.rwse_encoder(data.rwse.float())

        if edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr.float())
        else:
            edge_emb = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)

        for layer in self.layers:
            h = layer(h, edge_index, edge_emb, batch)

        h = self.final_norm(h)

        return h


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps graph embeddings to a lower-dimensional space where contrastive loss is computed.
    This helps learn better representations by separating the representation space
    from the contrastive learning space.

    Architecture: Linear → BatchNorm → ReLU → Linear → BatchNorm
    """
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        return self.net(x)


class FragmentGPSForPretraining(nn.Module):
    """
    GraphGPS model wrapper for contrastive pretraining

    Includes a projection head for contrastive learning.
    - GPS backbone: extracts graph-level embeddings
    - Projection head: maps embeddings to contrastive space

    During finetuning, only the GPS backbone is used (projection head is discarded).
    """
    def __init__(
        self,
        node_dim=768,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        attn_dropout=0.1,
        pe_dim=8,
        use_lap_pe=True,
        use_degree=True,
        use_rwse=False,
        rwse_dim=16,
        edge_dim=1,
        pool='mean',
        local_gnn_type='GINE',
        global_model_type='Transformer',
        max_degree=100,
        proj_hidden_dim=256,
        proj_out_dim=128
    ):
        super().__init__()
        self.gps = FragmentGPS(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            pe_dim=pe_dim,
            use_lap_pe=use_lap_pe,
            use_degree=use_degree,
            use_rwse=use_rwse,
            rwse_dim=rwse_dim,
            edge_dim=edge_dim,
            output_dim=None,
            pool=pool,
            local_gnn_type=local_gnn_type,
            global_model_type=global_model_type,
            max_degree=max_degree
        )

        self.projection_head = ProjectionHead(
            in_dim=hidden_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim
        )

    def forward(self, data, return_embedding=False):
        """
        Forward pass for contrastive pretraining

        Args:
            data: PyG batch data
            return_embedding: if True, return GPS embedding (before projection)
                              if False, return projected embedding (for contrastive loss)

        Returns:
            - return_embedding=False: projected embedding [B, proj_out_dim] for contrastive loss
            - return_embedding=True: GPS embedding [B, hidden_dim] for downstream tasks
        """
        embedding = self.gps(data)

        if return_embedding:
            return embedding

        projected = self.projection_head(embedding)

        return projected


class FragmentGPSForFinetuning(nn.Module):
    """
    GraphGPS model wrapper for finetuning (property prediction)
    """
    def __init__(
        self,
        node_dim=768,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        attn_dropout=0.1,
        pe_dim=8,
        use_lap_pe=True,
        use_degree=True,
        use_rwse=False,
        rwse_dim=16,
        edge_dim=1,
        output_dim=1,
        pool='mean',
        local_gnn_type='GINE',
        global_model_type='Transformer',
        max_degree=100
    ):
        super().__init__()
        self.gps = FragmentGPS(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            pe_dim=pe_dim,
            use_lap_pe=use_lap_pe,
            use_degree=use_degree,
            use_rwse=use_rwse,
            rwse_dim=rwse_dim,
            edge_dim=edge_dim,
            output_dim=output_dim,
            pool=pool,
            local_gnn_type=local_gnn_type,
            global_model_type=global_model_type,
            max_degree=max_degree
        )

    def forward(self, data):
        """
        Returns predictions for property tasks
        """
        return self.gps(data)

    def get_graph_embeddings(self, data):
        """
        Get graph-level embeddings (before output head)
        """
        return self.gps.gps(data)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, model_name="FragmentGPS"):
    """Print model architecture info"""
    print(f"\n{'='*60}")
    print(f"{model_name} Model Info")
    print(f"{'='*60}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"{'='*60}")
    for name, module in model.named_children():
        if hasattr(module, 'parameters'):
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {params:,} params")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    from torch_geometric.data import Data, Batch

    print("Testing FragmentGPS model...")

    num_nodes = 10
    num_edges = 20
    pe_dim = 8

    data1 = Data(
        x=torch.randn(num_nodes, 768),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 1),
        pe=torch.randn(num_nodes, pe_dim),
        in_degree=torch.randint(0, 10, (num_nodes,)),
        out_degree=torch.randint(0, 10, (num_nodes,))
    )

    data2 = Data(
        x=torch.randn(8, 768),
        edge_index=torch.randint(0, 8, (2, 15)),
        edge_attr=torch.randn(15, 1),
        pe=torch.randn(8, pe_dim),
        in_degree=torch.randint(0, 10, (8,)),
        out_degree=torch.randint(0, 10, (8,))
    )

    batch = Batch.from_data_list([data1, data2])

    model = FragmentGPSForPretraining(
        node_dim=768,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        pe_dim=pe_dim,
        use_lap_pe=True,
        use_degree=True
    )

    print_model_info(model, "FragmentGPSForPretraining")

    output = model(batch)
    print(f"Pretraining output shape: {output.shape}")

    model_ft = FragmentGPSForFinetuning(
        node_dim=768,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        pe_dim=pe_dim,
        output_dim=12
    )

    print_model_info(model_ft, "FragmentGPSForFinetuning")

    output_ft = model_ft(batch)
    print(f"Finetuning output shape: {output_ft.shape}")

    print("\nAll tests passed!")

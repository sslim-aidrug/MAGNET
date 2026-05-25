"""
GraphGPS Unified Pretraining Script (ChemBERTa-77M-MTR version)
Pretrain on ZINC250K with:
1. Graph-Graph Contrastive learning (InfoNCE, with node masking)
2. Self-supervised property prediction (RDKit descriptors)
3. SMILES-Graph Contrastive learning (cross-modal alignment)

SMILES Encoder: ChemBERTa-77M-MTR (DeepChem/ChemBERTa-77M-MTR)
Node Features: 549D (Morgan+PhysChem) + 384D (ChemBERTa-77M-MTR) = 933D
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

import pickle
import random
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, GraphDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import transformers
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from magnet.conf import get_conf
from magnet.gps_model import FragmentGPS, ProjectionHead


class ChemBERTa2Encoder(nn.Module):
    """ChemBERTa-77M-MTR SMILES encoder with projection head."""

    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MTR",
                 proj_hidden_dim=256, proj_out_dim=128, freeze_bert=True):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        print(f"Loading ChemBERTa-2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("ChemBERTa-2 weights frozen")

        self.projection_head = nn.Sequential(
            nn.Linear(self.bert_dim, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_out_dim)
        )
        print(f"ChemBERTa-2 Encoder: {self.bert_dim}D -> {proj_out_dim}D")

    def forward(self, smiles_list, device):
        inputs = self.tokenizer(smiles_list, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        projected = self.projection_head(embedding)
        return projected


class SMILESGraphContrastiveLoss(nn.Module):
    """Contrastive loss aligning SMILES and graph embeddings (cross-entropy over similarity)."""

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, smiles_emb, graph_emb):
        smiles_emb = F.normalize(smiles_emb, dim=1)
        graph_emb = F.normalize(graph_emb, dim=1)

        logits = torch.mm(smiles_emb, graph_emb.t()) / self.temperature
        batch_size = smiles_emb.size(0)
        labels = torch.arange(batch_size, device=smiles_emb.device)

        loss_s2g = F.cross_entropy(logits, labels)
        loss_g2s = F.cross_entropy(logits.t(), labels)
        return (loss_s2g + loss_g2s) / 2


def load_pickle(pkl_path):
    """Load a pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


PROPERTY_NAMES = [
    'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'NumSaturatedRings',
    'HeavyAtomCount', 'NumHeteroatoms', 'FractionCSP3',
    'BertzCT', 'HallKierAlpha',
    'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2',
    'LabuteASA'
]
NUM_PROPERTIES = len(PROPERTY_NAMES)


def compute_rdkit_descriptors(smiles):
    """
    Compute RDKit descriptors for a SMILES string.
    Returns a tensor of normalized descriptors (20 properties).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:

        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.BertzCT(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.LabuteASA(mol),
        ]
        return torch.tensor(descriptors, dtype=torch.float32)
    except:
        return None


def compute_descriptors_for_dataset(graphs, smiles_list=None, normalize=True):
    """
    Compute RDKit descriptors for all graphs in dataset.
    Adds 'rdkit_props' attribute to each graph.

    Args:
        graphs: List of graph objects
        smiles_list: List of SMILES strings (from filtered_data)
        normalize: Whether to normalize descriptors
    """
    print(f"Computing RDKit descriptors for {len(graphs)} molecules...")

    all_props = []
    valid_indices = []

    for i, g in enumerate(graphs):
        smiles = None
        if smiles_list is not None and i < len(smiles_list):
            smiles = smiles_list[i]
        elif hasattr(g, 'smiles') and g.smiles:
            smiles = g.smiles

        if smiles:
            props = compute_rdkit_descriptors(smiles)
            if props is not None:
                all_props.append(props)
                valid_indices.append(i)
                g.rdkit_props = props
            else:
                g.rdkit_props = torch.zeros(NUM_PROPERTIES)
        else:
            g.rdkit_props = torch.zeros(NUM_PROPERTIES)

    if normalize and len(all_props) > 0:
        all_props_tensor = torch.stack(all_props)
        mean = all_props_tensor.mean(dim=0)
        std = all_props_tensor.std(dim=0) + 1e-8

        print(f"Descriptor statistics:")
        for i, name in enumerate(PROPERTY_NAMES):
            print(f"  {name}: mean={mean[i]:.2f}, std={std[i]:.2f}")

        for g in graphs:
            g.rdkit_props = (g.rdkit_props - mean) / std

        return mean, std

    return None, None


class GPSForPretrainingWithProperty(nn.Module):
    """GPS model with Projection Head for contrastive + Property Prediction Head"""
    def __init__(
        self,
        node_dim=182,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        attn_dropout=0.1,
        pe_dim=8,
        use_lap_pe=True,
        use_degree=True,
        use_rwse=False,
        edge_dim=1,
        local_gnn_type='GATv2',
        proj_hidden_dim=256,
        proj_out_dim=128,
        num_properties=NUM_PROPERTIES
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
            edge_dim=edge_dim,
            output_dim=None,
            pool='mean',
            local_gnn_type=local_gnn_type,
            global_model_type='Transformer'
        )

        self.projection_head = ProjectionHead(
            in_dim=hidden_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim
        )

        self.smiles_graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_out_dim)
        )

        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_properties)
        )

        self.hidden_dim = hidden_dim

    def forward(self, data, return_property=False):
        embedding = self.gps(data)
        projected = self.projection_head(embedding)

        if return_property:
            property_pred = self.property_head(embedding)
            return projected, property_pred

        return projected

    def get_embedding(self, data):
        return self.gps(data)

    def get_smiles_graph_projection(self, data):
        """Get projection for SMILES-Graph contrastive learning."""
        embedding = self.gps(data)
        return self.smiles_graph_projection(embedding)

    def predict_property(self, data):
        embedding = self.gps(data)
        return self.property_head(embedding)


class MaskedGraphDataset:
    """Dataset wrapper for contrastive learning with masking augmentation"""
    def __init__(self, dataset, node_mask_ratio=0.2, smiles_list=None):
        self.dataset = dataset
        self.node_mask_ratio = node_mask_ratio
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.dataset)

    def mask_nodes(self, data):
        """Zero out a random fraction of node features (augmentation for contrastive views)."""
        x = data.x.clone()
        num_nodes = x.size(0)
        k = int(self.node_mask_ratio * num_nodes)
        if k > 0:
            mask_idx = random.sample(range(num_nodes), k)
            x[mask_idx] = 0
        data = data.clone()
        data.x = x
        return data

    def __getitem__(self, idx):
        original = self.dataset[idx]
        view1 = self.mask_nodes(original)
        view2 = self.mask_nodes(original)

        smiles = None
        if self.smiles_list is not None:
            if hasattr(self.dataset, 'indices'):
                actual_idx = self.dataset.indices[idx]
                smiles = self.smiles_list[actual_idx]
            else:
                smiles = self.smiles_list[idx]

        return original, view1, view2, smiles


def collate_pretrain(batch):
    """Collate (original, view1, view2, smiles) tuples into batched PyG graphs."""
    originals = [item[0] for item in batch]
    view1s = [item[1] for item in batch]
    view2s = [item[2] for item in batch]
    smiles_list = [item[3] for item in batch]
    return (Batch.from_data_list(originals),
            Batch.from_data_list(view1s),
            Batch.from_data_list(view2s),
            smiles_list)


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss between two augmented views."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size),
                           torch.arange(batch_size)]).to(z.device)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        return F.cross_entropy(sim, labels)


class PropertyPredictionLoss(nn.Module):
    """MSE loss for property prediction"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


def train_epoch(model, loader, optimizer, contrastive_loss_fn, property_loss_fn,
                device, epoch, property_weight=0.1,
                smiles_encoder=None, smiles_graph_loss_fn=None, smiles_graph_weight=0.1):
    """Train one epoch with combined contrastive + property + SMILES-Graph loss"""
    model.train()
    if smiles_encoder is not None:
        smiles_encoder.train()

    total_loss = 0
    total_contrastive = 0
    total_property = 0
    total_smiles_graph = 0
    num_batches = len(loader)

    disable_tqdm = not sys.stdout.isatty()
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=disable_tqdm)

    for batch in pbar:
        original, view1, view2, smiles_list = batch
        original = original.to(device)
        view1, view2 = view1.to(device), view2.to(device)

        # L_GG: graph-graph contrastive loss between two node-masked views
        z1 = model(view1)
        z2 = model(view2)
        contrastive_loss = contrastive_loss_fn(z1, z2)

        # L_P: regress normalized RDKit descriptors from the graph embedding
        property_pred = model.predict_property(original)
        property_target = original.rdkit_props.view(-1, NUM_PROPERTIES).to(device)
        property_loss = property_loss_fn(property_pred, property_target)

        # L_SG: align graph embedding with the frozen ChemBERTa SMILES embedding
        smiles_graph_loss = torch.tensor(0.0, device=device)
        if smiles_encoder is not None and smiles_graph_loss_fn is not None:
            valid_indices = [i for i, s in enumerate(smiles_list) if s is not None]
            if len(valid_indices) > 1:
                valid_smiles = [smiles_list[i] for i in valid_indices]

                smiles_emb = smiles_encoder(valid_smiles, device)

                valid_graphs = [original[i] for i in valid_indices]
                valid_batch = Batch.from_data_list(valid_graphs).to(device)
                graph_emb = model.get_smiles_graph_projection(valid_batch)

                smiles_graph_loss = smiles_graph_loss_fn(smiles_emb, graph_emb)

        # Multi-objective loss: L = L_GG + beta * L_P + gamma * L_SG
        loss = (contrastive_loss +
                property_weight * property_loss +
                smiles_graph_weight * smiles_graph_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_contrastive += contrastive_loss.item()
        total_property += property_loss.item()
        total_smiles_graph += smiles_graph_loss.item()

        if not disable_tqdm:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                           gg=f"{contrastive_loss.item():.4f}",
                           prop=f"{property_loss.item():.4f}",
                           sg=f"{smiles_graph_loss.item():.4f}")

    n = len(loader)
    return total_loss / n, total_contrastive / n, total_property / n, total_smiles_graph / n


@torch.no_grad()
def validate(model, loader, contrastive_loss_fn, property_loss_fn, device, property_weight=0.1,
             smiles_encoder=None, smiles_graph_loss_fn=None, smiles_graph_weight=0.1):
    """Validate with combined loss"""
    model.eval()
    if smiles_encoder is not None:
        smiles_encoder.eval()

    total_loss = 0
    total_contrastive = 0
    total_property = 0
    total_smiles_graph = 0

    for batch in loader:
        original, view1, view2, smiles_list = batch
        original = original.to(device)
        view1, view2 = view1.to(device), view2.to(device)

        # L_GG: graph-graph contrastive loss between two node-masked views
        z1 = model(view1)
        z2 = model(view2)
        contrastive_loss = contrastive_loss_fn(z1, z2)

        # L_P: regress normalized RDKit descriptors from the graph embedding
        property_pred = model.predict_property(original)
        property_target = original.rdkit_props.view(-1, NUM_PROPERTIES).to(device)
        property_loss = property_loss_fn(property_pred, property_target)

        # L_SG: align graph embedding with the frozen ChemBERTa SMILES embedding
        smiles_graph_loss = torch.tensor(0.0, device=device)
        if smiles_encoder is not None and smiles_graph_loss_fn is not None:
            valid_indices = [i for i, s in enumerate(smiles_list) if s is not None]
            if len(valid_indices) > 1:
                valid_smiles = [smiles_list[i] for i in valid_indices]
                smiles_emb = smiles_encoder(valid_smiles, device)
                valid_graphs = [original[i] for i in valid_indices]
                valid_batch = Batch.from_data_list(valid_graphs).to(device)
                graph_emb = model.get_smiles_graph_projection(valid_batch)
                smiles_graph_loss = smiles_graph_loss_fn(smiles_emb, graph_emb)

        # Multi-objective loss: L = L_GG + beta * L_P + gamma * L_SG
        loss = (contrastive_loss +
                property_weight * property_loss +
                smiles_graph_weight * smiles_graph_loss)

        total_loss += loss.item()
        total_contrastive += contrastive_loss.item()
        total_property += property_loss.item()
        total_smiles_graph += smiles_graph_loss.item()

    n = len(loader)
    return total_loss / n, total_contrastive / n, total_property / n, total_smiles_graph / n


def pretrain(model, train_loader, valid_loader, optimizer, scheduler,
             contrastive_loss_fn, property_loss_fn, device, epochs, patience,
             save_dir, property_weight=0.1,
             smiles_encoder=None, smiles_graph_loss_fn=None, smiles_graph_weight=0.1):
    """Full pretraining loop with contrastive + property prediction + SMILES-Graph alignment"""
    print("=" * 60)
    print("Unified Pretraining: Graph-Graph + Property + SMILES-Graph")
    print(f"Property Weight: {property_weight}")
    print(f"SMILES-Graph Weight: {smiles_graph_weight}")
    print("=" * 60)

    best_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_contr, train_prop, train_sg = train_epoch(
            model, train_loader, optimizer, contrastive_loss_fn, property_loss_fn,
            device, epoch, property_weight,
            smiles_encoder, smiles_graph_loss_fn, smiles_graph_weight
        )
        valid_loss, valid_contr, valid_prop, valid_sg = validate(
            model, valid_loader, contrastive_loss_fn, property_loss_fn,
            device, property_weight,
            smiles_encoder, smiles_graph_loss_fn, smiles_graph_weight
        )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} (GG:{train_contr:.4f}, P:{train_prop:.4f}, SG:{train_sg:.4f}) | "
              f"Valid: {valid_loss:.4f} (GG:{valid_contr:.4f}, P:{valid_prop:.4f}, SG:{valid_sg:.4f}) | LR: {lr:.2e}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            print(f"  -> New best model (valid_loss: {valid_loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch in [50, 75, 100] or epoch % 50 == 0:
            gps_state = {k: v for k, v in model.state_dict().items() if k.startswith('gps.')}
            gps_state = {k[4:]: v for k, v in gps_state.items()}
            torch.save(gps_state, save_dir / f"pretrained_gps_epoch{epoch}.pt")
            print(f"  -> Saved intermediate checkpoint: epoch {epoch}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    save_path = save_dir / "pretrained_gps.pt"
    gps_state = {k: v for k, v in model.state_dict().items() if k.startswith('gps.')}
    gps_state = {k[4:]: v for k, v in gps_state.items()}
    torch.save(gps_state, save_path)
    print(f"\nPretrained GPS saved to: {save_path}")
    print(f"Best valid loss: {best_loss:.4f}")

    return model


def main():
    """Pre-train MAGNET on ZINC250K with the multi-objective loss (L_GG + L_P + L_SG)."""
    args = get_conf()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    random.seed(args.seeds)
    np.random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    torch.cuda.manual_seed_all(args.seeds)

    property_weight = args.property_weight

    print(f"\nLoading pretraining data from: {args.pre_graph_pkl}")
    pre_data = load_pickle(args.pre_graph_pkl)
    pre_graphs = pre_data["zinc250k_all_graphs"]
    print(f"Total graphs: {len(pre_graphs)}")

    smiles_list = None
    for key in pre_data.keys():
        if 'filtered_data' in key:
            filtered_data = pre_data[key]
            if hasattr(filtered_data, 'columns') and 'smiles' in filtered_data.columns:
                smiles_list = filtered_data['smiles'].tolist()
                print(f"Loaded {len(smiles_list)} SMILES from {key}")
            break

    if smiles_list is None:
        print("WARNING: No SMILES found in filtered_data, trying graph attributes")

    mean, std = compute_descriptors_for_dataset(pre_graphs, smiles_list=smiles_list, normalize=True)

    save_dir = Path(args.base_dir) / "pretrain_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    if mean is not None:
        torch.save({'mean': mean, 'std': std}, save_dir / "rdkit_norm_stats.pt")

    n_valid = int(0.1 * len(pre_graphs))
    n_train = len(pre_graphs) - n_valid
    pre_train, pre_valid = torch.utils.data.random_split(
        pre_graphs, [n_train, n_valid],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {len(pre_train)}, Valid: {len(pre_valid)}")

    train_dataset = MaskedGraphDataset(pre_train, args.node_mask_ratio, smiles_list)
    valid_dataset = MaskedGraphDataset(pre_valid, args.node_mask_ratio, smiles_list)

    train_loader = PyGDataLoader(train_dataset, batch_size=args.pre_batch_size,
                                  shuffle=True, collate_fn=collate_pretrain, num_workers=4)
    valid_loader = PyGDataLoader(valid_dataset, batch_size=args.pre_batch_size,
                                  shuffle=False, collate_fn=collate_pretrain, num_workers=4)

    model = GPSForPretrainingWithProperty(
        node_dim=args.node_dim,
        hidden_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        pe_dim=args.pe_dim,
        use_lap_pe=args.use_lap_pe,
        use_degree=args.use_degree,
        use_rwse=args.use_rwse,
        edge_dim=args.edge_dim,
        local_gnn_type=args.local_gnn_type,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
        num_properties=NUM_PROPERTIES
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Properties: {NUM_PROPERTIES} ({PROPERTY_NAMES})")

    smiles_graph_weight = args.smiles_graph_weight
    smiles_graph_temperature = getattr(args, 'smiles_graph_temperature', 0.2)

    smiles_encoder = None
    smiles_graph_loss_fn = None
    if smiles_list is not None:
        print(f"\nInitializing ChemBERTa-2 encoder for SMILES-Graph contrastive...")
        smiles_encoder = ChemBERTa2Encoder(
            model_name="DeepChem/ChemBERTa-77M-MTR",
            proj_hidden_dim=args.proj_hidden_dim,
            proj_out_dim=args.proj_out_dim,
            freeze_bert=True
        ).to(device)
        smiles_graph_loss_fn = SMILESGraphContrastiveLoss(temperature=smiles_graph_temperature)
        print(f"SMILES-Graph Weight: {smiles_graph_weight}, Temperature: {smiles_graph_temperature}")

    all_params = list(model.parameters())
    if smiles_encoder is not None:
        all_params += list(smiles_encoder.projection_head.parameters())

    optimizer = optim.AdamW(all_params, lr=args.pre_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.pre_epochs, eta_min=1e-6)
    contrastive_loss_fn = ContrastiveLoss(temperature=args.temperature)
    property_loss_fn = PropertyPredictionLoss()

    model = pretrain(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        contrastive_loss_fn=contrastive_loss_fn,
        property_loss_fn=property_loss_fn,
        device=device,
        epochs=args.pre_epochs,
        patience=args.pre_patience,
        save_dir=save_dir,
        property_weight=property_weight,
        smiles_encoder=smiles_encoder,
        smiles_graph_loss_fn=smiles_graph_loss_fn,
        smiles_graph_weight=smiles_graph_weight
    )

    print("\n" + "=" * 60)
    print("Pretraining completed!")
    print(f"Model saved to: {save_dir / 'pretrained_gps.pt'}")
    print("=" * 60)


if __name__ == '__main__':
    main()

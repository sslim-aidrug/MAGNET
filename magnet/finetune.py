"""
Unified GraphGPS Finetuning Script - ChemBERTa-2 Version (Frozen) with Checkpoint Saving

Same as finetune_unified_chemberta2.py but saves the best run's checkpoint.
After all runs complete, saves the model state_dict from the run with the best test score.

Usage:
    python finetune_unified_chemberta2_ckpt.py --dataset-name BBBP --smiles-feature-type concat_all \
        --node-dim 933 --chemberta-model-name DeepChem/ChemBERTa-77M-MTR --pretrained-path <path> \
        --save-checkpoint-dir <dir>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List
import os

import pickle
import random
import numpy as np
import sys

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from magnet.conf import get_conf
from magnet.gps_model import FragmentGPS

import warnings
warnings.filterwarnings('ignore')


SMILES_DIM_MAP = {
    'concat_all': 933,
}


# --- SMILES feature embedders (concatenated with the graph embedding at fine-tuning) ---
class FullConcatEmbedder:
    """Full concat embedding: 549D Morgan + ChemBERTa (dynamic) = dynamic total"""

    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", device="cuda"):
        self.device = device
        self.morgan_dim = 549
        from transformers import AutoTokenizer, AutoModel
        from magnet.metagraph.node_features import compute_enhanced_features
        self.compute_features = compute_enhanced_features
        print(f"Loading ChemBERTa model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.chemberta_dim = self.model.config.hidden_size
        self.embedding_dim = self.morgan_dim + self.chemberta_dim
        print(f"Full Concat embedding: Morgan {self.morgan_dim}D + ChemBERTa {self.chemberta_dim}D = {self.embedding_dim}D")

    @torch.no_grad()
    def get_batch_embeddings(self, smiles_list: List[str], batch_size: int = 64) -> torch.Tensor:
        """Get full concat embeddings for a batch of SMILES strings."""
        all_embeddings = []
        disable_tqdm = not sys.stdout.isatty()

        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Full Concat encoding", disable=disable_tqdm):
            batch_smiles = smiles_list[i:i+batch_size]

            inputs = self.tokenizer(batch_smiles, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            chemberta_emb = outputs.last_hidden_state[:, 0, :].cpu()

            morgan_features = []
            for smi in batch_smiles:
                feat = self.compute_features(smi)
                morgan_features.append(feat)
            morgan_emb = torch.tensor(morgan_features, dtype=torch.float32)

            concat_emb = torch.cat([morgan_emb, chemberta_emb], dim=1)
            all_embeddings.append(concat_emb)

        return torch.cat(all_embeddings, dim=0)

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


def get_embedder(smiles_type: str, model_name: str, device: str):
    """Factory function to create appropriate embedder."""
    if smiles_type == 'concat_all':
        return FullConcatEmbedder(model_name, device)
    else:
        raise ValueError(f"Unknown smiles_feature_type: {smiles_type}")


def add_embeddings_to_graphs(graphs, smiles_list, embedder, batch_size=64):
    """Add SMILES embeddings to each graph as a graph-level attribute."""
    print(f"Generating {embedder.__class__.__name__} embeddings...")

    all_embeddings = embedder.get_batch_embeddings(smiles_list, batch_size=batch_size)
    for i, graph in enumerate(graphs):
        graph.smiles_emb = all_embeddings[i].unsqueeze(0)

    return graphs


class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch,
                 init_lr, max_lr, final_lr):
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == \
               len(init_lr) == len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr.copy() if isinstance(init_lr, list) else list(init_lr)
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / np.maximum(self.warmup_steps, 1)
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / np.maximum(self.total_steps - self.warmup_steps, 1))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        return list(self.lr)

    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:
                self.lr[i] = self.final_lr[i]
            self.optimizer.param_groups[i]['lr'] = self.lr[i]


class GPSForFinetuning(nn.Module):
    """
    GPS encoder fused with SMILES features via cross-attention.
    The graph embedding (256D) is combined with concatenated Morgan+ChemBERTa
    SMILES features (933D) for a 1189D prediction-head input.
    """

    def __init__(self, node_dim=182, hidden_dim=256, smiles_dim=0, num_layers=2,
                 num_heads=4, dropout=0.1, attn_dropout=0.1, pe_dim=8,
                 use_lap_pe=True, use_degree=True, use_rwse=False, edge_dim=1,
                 local_gnn_type='GATv2', output_dim=1, task_type='classification',
                 smiles_proj_dim=0):
        super().__init__()

        self.smiles_dim = smiles_dim
        self.smiles_proj_dim = smiles_proj_dim

        self.gps = FragmentGPS(
            node_dim=node_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, attn_dropout=attn_dropout,
            pe_dim=pe_dim, use_lap_pe=use_lap_pe, use_degree=use_degree,
            use_rwse=use_rwse, edge_dim=edge_dim, output_dim=None,
            pool='mean', local_gnn_type=local_gnn_type, global_model_type='Transformer'
        )

        if smiles_proj_dim > 0:
            self.smiles_projection = nn.Sequential(
                nn.Linear(smiles_dim, smiles_proj_dim),
                nn.LayerNorm(smiles_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            smiles_out_dim = smiles_proj_dim
        else:
            self.smiles_projection = None
            smiles_out_dim = smiles_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_ln = nn.LayerNorm(hidden_dim)
        if smiles_out_dim != hidden_dim:
            self.smiles_to_hidden = nn.Linear(smiles_out_dim, hidden_dim)
        else:
            self.smiles_to_hidden = None
        combined_dim = hidden_dim + smiles_out_dim


        self.hidden_dim = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        graph_emb = self.gps(data)

        smiles_emb = data.smiles_emb
        if smiles_emb.dim() == 1:
            smiles_emb = smiles_emb.view(-1, self.smiles_dim)
        if self.smiles_projection is not None:
            smiles_emb = self.smiles_projection(smiles_emb)

        if self.smiles_to_hidden is not None:
            smiles_for_attn = self.smiles_to_hidden(smiles_emb)
        else:
            smiles_for_attn = smiles_emb

        graph_q = graph_emb.unsqueeze(1)
        smiles_kv = smiles_for_attn.unsqueeze(1)

        attended, _ = self.cross_attn(graph_q, smiles_kv, smiles_kv)
        attended = attended.squeeze(1)

        attended = self.cross_attn_ln(graph_emb + attended)

        combined = torch.cat([attended, smiles_emb], dim=1)

        output = self.head(combined)


        return output

    def load_pretrained_gps(self, pretrained_path, device):
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
        self.gps.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained GPS from: {pretrained_path}")

    def freeze_gps(self):
        """Freeze GPS parameters for transfer learning."""
        for param in self.gps.parameters():
            param.requires_grad = False

    def unfreeze_gps(self):
        """Unfreeze GPS parameters."""
        for param in self.gps.parameters():
            param.requires_grad = True


def load_pickle(pkl_path):
    """Load a pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def load_split_indices(dataset_name, base_dir, split_type, seed):
    """Load precomputed train/val/test indices from {dataset}-{split_type}-{seed}.npz."""
    npz_path = Path(base_dir) / dataset_name / f"{dataset_name}-{split_type}-{seed}.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    tr, va, te = data['train'], data['val'], data['test']

    def fix_array(a):
        a = np.asarray(a).reshape(-1)
        if np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.int64)
        return a

    return fix_array(tr), fix_array(va), fix_array(te)


def get_labels_by_dataset(dataset_name, filtered_data):
    """Extract the label column(s) for a dataset from its filtered DataFrame."""
    dataset_key = dataset_name.lower()
    label_mapping = {
        'bbbp': 'p_np', 'bace': 'Class', 'hiv': 'HIV_active',
        'esol': 'measured log solubility in mols per litre',
        'freesolv': 'expt', 'lipo': 'exp',
    }
    multi_label_mapping = {
        'sider': ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
                  "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
                  "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
                  "Reproductive system and breast disorders",
                  "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                  "General disorders and administration site conditions", "Endocrine disorders",
                  "Surgical and medical procedures", "Vascular disorders",
                  "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
                  "Congenital, familial and genetic disorders", "Infections and infestations",
                  "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
                  "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
                  "Ear and labyrinth disorders", "Cardiac disorders",
                  "Nervous system disorders", "Injury, poisoning and procedural complications"],
        'clintox': ['CT_TOX', 'FDA_APPROVED'],
        'tox21': ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                  "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"],
        'muv': ["MUV-692", "MUV-689", "MUV-846", "MUV-859", "MUV-644", "MUV-548", "MUV-852",
                "MUV-600", "MUV-810", "MUV-712", "MUV-737", "MUV-858", "MUV-713", "MUV-733",
                "MUV-652", "MUV-466", "MUV-832"],
        'toxcast': ["ACEA_T47D_80hr_Negative","ACEA_T47D_80hr_Positive","APR_HepG2_CellCycleArrest_24h_dn","APR_HepG2_CellCycleArrest_24h_up","APR_HepG2_CellCycleArrest_72h_dn","APR_HepG2_CellLoss_24h_dn","APR_HepG2_CellLoss_72h_dn","APR_HepG2_MicrotubuleCSK_24h_dn","APR_HepG2_MicrotubuleCSK_24h_up","APR_HepG2_MicrotubuleCSK_72h_dn","APR_HepG2_MicrotubuleCSK_72h_up","APR_HepG2_MitoMass_24h_dn","APR_HepG2_MitoMass_24h_up","APR_HepG2_MitoMass_72h_dn","APR_HepG2_MitoMass_72h_up","APR_HepG2_MitoMembPot_1h_dn","APR_HepG2_MitoMembPot_24h_dn","APR_HepG2_MitoMembPot_72h_dn","APR_HepG2_MitoticArrest_24h_up","APR_HepG2_MitoticArrest_72h_up","APR_HepG2_NuclearSize_24h_dn","APR_HepG2_NuclearSize_72h_dn","APR_HepG2_NuclearSize_72h_up","APR_HepG2_OxidativeStress_24h_up","APR_HepG2_OxidativeStress_72h_up","APR_HepG2_StressKinase_1h_up","APR_HepG2_StressKinase_24h_up","APR_HepG2_StressKinase_72h_up","APR_HepG2_p53Act_24h_up","APR_HepG2_p53Act_72h_up","APR_Hepat_Apoptosis_24hr_up","APR_Hepat_Apoptosis_48hr_up","APR_Hepat_CellLoss_24hr_dn","APR_Hepat_CellLoss_48hr_dn","APR_Hepat_DNADamage_24hr_up","APR_Hepat_DNADamage_48hr_up","APR_Hepat_DNATexture_24hr_up","APR_Hepat_DNATexture_48hr_up","APR_Hepat_MitoFxnI_1hr_dn","APR_Hepat_MitoFxnI_24hr_dn","APR_Hepat_MitoFxnI_48hr_dn","APR_Hepat_NuclearSize_24hr_dn","APR_Hepat_NuclearSize_48hr_dn","APR_Hepat_Steatosis_24hr_up","APR_Hepat_Steatosis_48hr_up","ATG_AP_1_CIS_dn","ATG_AP_1_CIS_up","ATG_AP_2_CIS_dn","ATG_AP_2_CIS_up","ATG_AR_TRANS_dn","ATG_AR_TRANS_up","ATG_Ahr_CIS_dn","ATG_Ahr_CIS_up","ATG_BRE_CIS_dn","ATG_BRE_CIS_up","ATG_CAR_TRANS_dn","ATG_CAR_TRANS_up","ATG_CMV_CIS_dn","ATG_CMV_CIS_up","ATG_CRE_CIS_dn","ATG_CRE_CIS_up","ATG_C_EBP_CIS_dn","ATG_C_EBP_CIS_up","ATG_DR4_LXR_CIS_dn","ATG_DR4_LXR_CIS_up","ATG_DR5_CIS_dn","ATG_DR5_CIS_up","ATG_E2F_CIS_dn","ATG_E2F_CIS_up","ATG_EGR_CIS_up","ATG_ERE_CIS_dn","ATG_ERE_CIS_up","ATG_ERRa_TRANS_dn","ATG_ERRg_TRANS_dn","ATG_ERRg_TRANS_up","ATG_ERa_TRANS_up","ATG_E_Box_CIS_dn","ATG_E_Box_CIS_up","ATG_Ets_CIS_dn","ATG_Ets_CIS_up","ATG_FXR_TRANS_up","ATG_FoxA2_CIS_dn","ATG_FoxA2_CIS_up","ATG_FoxO_CIS_dn","ATG_FoxO_CIS_up","ATG_GAL4_TRANS_dn","ATG_GATA_CIS_dn","ATG_GATA_CIS_up","ATG_GLI_CIS_dn","ATG_GLI_CIS_up","ATG_GRE_CIS_dn","ATG_GRE_CIS_up","ATG_GR_TRANS_dn","ATG_GR_TRANS_up","ATG_HIF1a_CIS_dn","ATG_HIF1a_CIS_up","ATG_HNF4a_TRANS_dn","ATG_HNF4a_TRANS_up","ATG_HNF6_CIS_dn","ATG_HNF6_CIS_up","ATG_HSE_CIS_dn","ATG_HSE_CIS_up","ATG_IR1_CIS_dn","ATG_IR1_CIS_up","ATG_ISRE_CIS_dn","ATG_ISRE_CIS_up","ATG_LXRa_TRANS_dn","ATG_LXRa_TRANS_up","ATG_LXRb_TRANS_dn","ATG_LXRb_TRANS_up","ATG_MRE_CIS_up","ATG_M_06_TRANS_up","ATG_M_19_CIS_dn","ATG_M_19_TRANS_dn","ATG_M_19_TRANS_up","ATG_M_32_CIS_dn","ATG_M_32_CIS_up","ATG_M_32_TRANS_dn","ATG_M_32_TRANS_up","ATG_M_61_TRANS_up","ATG_Myb_CIS_dn","ATG_Myb_CIS_up","ATG_Myc_CIS_dn","ATG_Myc_CIS_up","ATG_NFI_CIS_dn","ATG_NFI_CIS_up","ATG_NF_kB_CIS_dn","ATG_NF_kB_CIS_up","ATG_NRF1_CIS_dn","ATG_NRF1_CIS_up","ATG_NRF2_ARE_CIS_dn","ATG_NRF2_ARE_CIS_up","ATG_NURR1_TRANS_dn","ATG_NURR1_TRANS_up","ATG_Oct_MLP_CIS_dn","ATG_Oct_MLP_CIS_up","ATG_PBREM_CIS_dn","ATG_PBREM_CIS_up","ATG_PPARa_TRANS_dn","ATG_PPARa_TRANS_up","ATG_PPARd_TRANS_up","ATG_PPARg_TRANS_up","ATG_PPRE_CIS_dn","ATG_PPRE_CIS_up","ATG_PXRE_CIS_dn","ATG_PXRE_CIS_up","ATG_PXR_TRANS_dn","ATG_PXR_TRANS_up","ATG_Pax6_CIS_up","ATG_RARa_TRANS_dn","ATG_RARa_TRANS_up","ATG_RARb_TRANS_dn","ATG_RARb_TRANS_up","ATG_RARg_TRANS_dn","ATG_RARg_TRANS_up","ATG_RORE_CIS_dn","ATG_RORE_CIS_up","ATG_RORb_TRANS_dn","ATG_RORg_TRANS_dn","ATG_RORg_TRANS_up","ATG_RXRa_TRANS_dn","ATG_RXRa_TRANS_up","ATG_RXRb_TRANS_dn","ATG_RXRb_TRANS_up","ATG_SREBP_CIS_dn","ATG_SREBP_CIS_up","ATG_STAT3_CIS_dn","ATG_STAT3_CIS_up","ATG_Sox_CIS_dn","ATG_Sox_CIS_up","ATG_Sp1_CIS_dn","ATG_Sp1_CIS_up","ATG_TAL_CIS_dn","ATG_TAL_CIS_up","ATG_TA_CIS_dn","ATG_TA_CIS_up","ATG_TCF_b_cat_CIS_dn","ATG_TCF_b_cat_CIS_up","ATG_TGFb_CIS_dn","ATG_TGFb_CIS_up","ATG_THRa1_TRANS_dn","ATG_THRa1_TRANS_up","ATG_VDRE_CIS_dn","ATG_VDRE_CIS_up","ATG_VDR_TRANS_dn","ATG_VDR_TRANS_up","ATG_XTT_Cytotoxicity_up","ATG_Xbp1_CIS_dn","ATG_Xbp1_CIS_up","ATG_p53_CIS_dn","ATG_p53_CIS_up","BSK_3C_Eselectin_down","BSK_3C_HLADR_down","BSK_3C_ICAM1_down","BSK_3C_IL8_down","BSK_3C_MCP1_down","BSK_3C_MIG_down","BSK_3C_Proliferation_down","BSK_3C_SRB_down","BSK_3C_Thrombomodulin_down","BSK_3C_Thrombomodulin_up","BSK_3C_TissueFactor_down","BSK_3C_TissueFactor_up","BSK_3C_VCAM1_down","BSK_3C_Vis_down","BSK_3C_uPAR_down","BSK_4H_Eotaxin3_down","BSK_4H_MCP1_down","BSK_4H_Pselectin_down","BSK_4H_Pselectin_up","BSK_4H_SRB_down","BSK_4H_VCAM1_down","BSK_4H_VEGFRII_down","BSK_4H_uPAR_down","BSK_4H_uPAR_up","BSK_BE3C_HLADR_down","BSK_BE3C_IL1a_down","BSK_BE3C_IP10_down","BSK_BE3C_MIG_down","BSK_BE3C_MMP1_down","BSK_BE3C_MMP1_up","BSK_BE3C_PAI1_down","BSK_BE3C_SRB_down","BSK_BE3C_TGFb1_down","BSK_BE3C_tPA_down","BSK_BE3C_uPAR_down","BSK_BE3C_uPAR_up","BSK_BE3C_uPA_down","BSK_CASM3C_HLADR_down","BSK_CASM3C_IL6_down","BSK_CASM3C_IL6_up","BSK_CASM3C_IL8_down","BSK_CASM3C_LDLR_down","BSK_CASM3C_LDLR_up","BSK_CASM3C_MCP1_down","BSK_CASM3C_MCP1_up","BSK_CASM3C_MCSF_down","BSK_CASM3C_MCSF_up","BSK_CASM3C_MIG_down","BSK_CASM3C_Proliferation_down","BSK_CASM3C_Proliferation_up","BSK_CASM3C_SAA_down","BSK_CASM3C_SAA_up","BSK_CASM3C_SRB_down","BSK_CASM3C_Thrombomodulin_down","BSK_CASM3C_Thrombomodulin_up","BSK_CASM3C_TissueFactor_down","BSK_CASM3C_VCAM1_down","BSK_CASM3C_VCAM1_up","BSK_CASM3C_uPAR_down","BSK_CASM3C_uPAR_up","BSK_KF3CT_ICAM1_down","BSK_KF3CT_IL1a_down","BSK_KF3CT_IP10_down","BSK_KF3CT_IP10_up","BSK_KF3CT_MCP1_down","BSK_KF3CT_MCP1_up","BSK_KF3CT_MMP9_down","BSK_KF3CT_SRB_down","BSK_KF3CT_TGFb1_down","BSK_KF3CT_TIMP2_down","BSK_KF3CT_uPA_down","BSK_LPS_CD40_down","BSK_LPS_Eselectin_down","BSK_LPS_Eselectin_up","BSK_LPS_IL1a_down","BSK_LPS_IL1a_up","BSK_LPS_IL8_down","BSK_LPS_IL8_up","BSK_LPS_MCP1_down","BSK_LPS_MCSF_down","BSK_LPS_PGE2_down","BSK_LPS_PGE2_up","BSK_LPS_SRB_down","BSK_LPS_TNFa_down","BSK_LPS_TNFa_up","BSK_LPS_TissueFactor_down","BSK_LPS_TissueFactor_up","BSK_LPS_VCAM1_down","BSK_SAg_CD38_down","BSK_SAg_CD40_down","BSK_SAg_CD69_down","BSK_SAg_Eselectin_down","BSK_SAg_Eselectin_up","BSK_SAg_IL8_down","BSK_SAg_IL8_up","BSK_SAg_MCP1_down","BSK_SAg_MIG_down","BSK_SAg_PBMCCytotoxicity_down","BSK_SAg_PBMCCytotoxicity_up","BSK_SAg_Proliferation_down","BSK_SAg_SRB_down","BSK_hDFCGF_CollagenIII_down","BSK_hDFCGF_EGFR_down","BSK_hDFCGF_EGFR_up","BSK_hDFCGF_IL8_down","BSK_hDFCGF_IP10_down","BSK_hDFCGF_MCSF_down","BSK_hDFCGF_MIG_down","BSK_hDFCGF_MMP1_down","BSK_hDFCGF_MMP1_up","BSK_hDFCGF_PAI1_down","BSK_hDFCGF_Proliferation_down","BSK_hDFCGF_SRB_down","BSK_hDFCGF_TIMP1_down","BSK_hDFCGF_VCAM1_down","CEETOX_H295R_11DCORT_dn","CEETOX_H295R_ANDR_dn","CEETOX_H295R_CORTISOL_dn","CEETOX_H295R_DOC_dn","CEETOX_H295R_DOC_up","CEETOX_H295R_ESTRADIOL_dn","CEETOX_H295R_ESTRADIOL_up","CEETOX_H295R_ESTRONE_dn","CEETOX_H295R_ESTRONE_up","CEETOX_H295R_OHPREG_up","CEETOX_H295R_OHPROG_dn","CEETOX_H295R_OHPROG_up","CEETOX_H295R_PROG_up","CEETOX_H295R_TESTO_dn","CLD_ABCB1_48hr","CLD_ABCG2_48hr","CLD_CYP1A1_24hr","CLD_CYP1A1_48hr","CLD_CYP1A1_6hr","CLD_CYP1A2_24hr","CLD_CYP1A2_48hr","CLD_CYP1A2_6hr","CLD_CYP2B6_24hr","CLD_CYP2B6_48hr","CLD_CYP2B6_6hr","CLD_CYP3A4_24hr","CLD_CYP3A4_48hr","CLD_CYP3A4_6hr","CLD_GSTA2_48hr","CLD_SULT2A_24hr","CLD_SULT2A_48hr","CLD_UGT1A1_24hr","CLD_UGT1A1_48hr","NCCT_HEK293T_CellTiterGLO","NCCT_QuantiLum_inhib_2_dn","NCCT_QuantiLum_inhib_dn","NCCT_TPO_AUR_dn","NCCT_TPO_GUA_dn","NHEERL_ZF_144hpf_TERATOSCORE_up","NVS_ADME_hCYP19A1","NVS_ADME_hCYP1A1","NVS_ADME_hCYP1A2","NVS_ADME_hCYP2A6","NVS_ADME_hCYP2B6","NVS_ADME_hCYP2C19","NVS_ADME_hCYP2C9","NVS_ADME_hCYP2D6","NVS_ADME_hCYP3A4","NVS_ADME_hCYP4F12","NVS_ADME_rCYP2C12","NVS_ENZ_hAChE","NVS_ENZ_hAMPKa1","NVS_ENZ_hAurA","NVS_ENZ_hBACE","NVS_ENZ_hCASP5","NVS_ENZ_hCK1D","NVS_ENZ_hDUSP3","NVS_ENZ_hES","NVS_ENZ_hElastase","NVS_ENZ_hFGFR1","NVS_ENZ_hGSK3b","NVS_ENZ_hMMP1","NVS_ENZ_hMMP13","NVS_ENZ_hMMP2","NVS_ENZ_hMMP3","NVS_ENZ_hMMP7","NVS_ENZ_hMMP9","NVS_ENZ_hPDE10","NVS_ENZ_hPDE4A1","NVS_ENZ_hPDE5","NVS_ENZ_hPI3Ka","NVS_ENZ_hPTEN","NVS_ENZ_hPTPN11","NVS_ENZ_hPTPN12","NVS_ENZ_hPTPN13","NVS_ENZ_hPTPN9","NVS_ENZ_hPTPRC","NVS_ENZ_hSIRT1","NVS_ENZ_hSIRT2","NVS_ENZ_hTrkA","NVS_ENZ_hVEGFR2","NVS_ENZ_oCOX1","NVS_ENZ_oCOX2","NVS_ENZ_rAChE","NVS_ENZ_rCNOS","NVS_ENZ_rMAOAC","NVS_ENZ_rMAOAP","NVS_ENZ_rMAOBC","NVS_ENZ_rMAOBP","NVS_ENZ_rabI2C","NVS_GPCR_bAdoR_NonSelective","NVS_GPCR_bDR_NonSelective","NVS_GPCR_g5HT4","NVS_GPCR_gH2","NVS_GPCR_gLTB4","NVS_GPCR_gLTD4","NVS_GPCR_gMPeripheral_NonSelective","NVS_GPCR_gOpiateK","NVS_GPCR_h5HT2A","NVS_GPCR_h5HT5A","NVS_GPCR_h5HT6","NVS_GPCR_h5HT7","NVS_GPCR_hAT1","NVS_GPCR_hAdoRA1","NVS_GPCR_hAdoRA2a","NVS_GPCR_hAdra2A","NVS_GPCR_hAdra2C","NVS_GPCR_hAdrb1","NVS_GPCR_hAdrb2","NVS_GPCR_hAdrb3","NVS_GPCR_hDRD1","NVS_GPCR_hDRD2s","NVS_GPCR_hDRD4.4","NVS_GPCR_hH1","NVS_GPCR_hLTB4_BLT1","NVS_GPCR_hM1","NVS_GPCR_hM2","NVS_GPCR_hM3","NVS_GPCR_hM4","NVS_GPCR_hNK2","NVS_GPCR_hOpiate_D1","NVS_GPCR_hOpiate_mu","NVS_GPCR_hTXA2","NVS_GPCR_p5HT2C","NVS_GPCR_r5HT1_NonSelective","NVS_GPCR_r5HT_NonSelective","NVS_GPCR_rAdra1B","NVS_GPCR_rAdra1_NonSelective","NVS_GPCR_rAdra2_NonSelective","NVS_GPCR_rAdrb_NonSelective","NVS_GPCR_rNK1","NVS_GPCR_rNK3","NVS_GPCR_rOpiate_NonSelective","NVS_GPCR_rOpiate_NonSelectiveNa","NVS_GPCR_rSST","NVS_GPCR_rTRH","NVS_GPCR_rV1","NVS_GPCR_rabPAF","NVS_GPCR_rmAdra2B","NVS_IC_hKhERGCh","NVS_IC_rCaBTZCHL","NVS_IC_rCaDHPRCh_L","NVS_IC_rNaCh_site2","NVS_LGIC_bGABARa1","NVS_LGIC_h5HT3","NVS_LGIC_hNNR_NBungSens","NVS_LGIC_rGABAR_NonSelective","NVS_LGIC_rNNR_BungSens","NVS_MP_hPBR","NVS_MP_rPBR","NVS_NR_bER","NVS_NR_bPR","NVS_NR_cAR","NVS_NR_hAR","NVS_NR_hCAR_Antagonist","NVS_NR_hER","NVS_NR_hFXR_Agonist","NVS_NR_hFXR_Antagonist","NVS_NR_hGR","NVS_NR_hPPARa","NVS_NR_hPPARg","NVS_NR_hPR","NVS_NR_hPXR","NVS_NR_hRAR_Antagonist","NVS_NR_hRARa_Agonist","NVS_NR_hTRa_Antagonist","NVS_NR_mERa","NVS_NR_rAR","NVS_NR_rMR","NVS_OR_gSIGMA_NonSelective","NVS_TR_gDAT","NVS_TR_hAdoT","NVS_TR_hDAT","NVS_TR_hNET","NVS_TR_hSERT","NVS_TR_rNET","NVS_TR_rSERT","NVS_TR_rVMAT2","OT_AR_ARELUC_AG_1440","OT_AR_ARSRC1_0480","OT_AR_ARSRC1_0960","OT_ER_ERaERa_0480","OT_ER_ERaERa_1440","OT_ER_ERaERb_0480","OT_ER_ERaERb_1440","OT_ER_ERbERb_0480","OT_ER_ERbERb_1440","OT_ERa_EREGFP_0120","OT_ERa_EREGFP_0480","OT_FXR_FXRSRC1_0480","OT_FXR_FXRSRC1_1440","OT_NURR1_NURR1RXRa_0480","OT_NURR1_NURR1RXRa_1440","TOX21_ARE_BLA_Agonist_ch1","TOX21_ARE_BLA_Agonist_ch2","TOX21_ARE_BLA_agonist_ratio","TOX21_ARE_BLA_agonist_viability","TOX21_AR_BLA_Agonist_ch1","TOX21_AR_BLA_Agonist_ch2","TOX21_AR_BLA_Agonist_ratio","TOX21_AR_BLA_Antagonist_ch1","TOX21_AR_BLA_Antagonist_ch2","TOX21_AR_BLA_Antagonist_ratio","TOX21_AR_BLA_Antagonist_viability","TOX21_AR_LUC_MDAKB2_Agonist","TOX21_AR_LUC_MDAKB2_Antagonist","TOX21_AR_LUC_MDAKB2_Antagonist2","TOX21_AhR_LUC_Agonist","TOX21_Aromatase_Inhibition","TOX21_AutoFluor_HEK293_Cell_blue","TOX21_AutoFluor_HEK293_Media_blue","TOX21_AutoFluor_HEPG2_Cell_blue","TOX21_AutoFluor_HEPG2_Cell_green","TOX21_AutoFluor_HEPG2_Media_blue","TOX21_AutoFluor_HEPG2_Media_green","TOX21_ELG1_LUC_Agonist","TOX21_ERa_BLA_Agonist_ch1","TOX21_ERa_BLA_Agonist_ch2","TOX21_ERa_BLA_Agonist_ratio","TOX21_ERa_BLA_Antagonist_ch1","TOX21_ERa_BLA_Antagonist_ch2","TOX21_ERa_BLA_Antagonist_ratio","TOX21_ERa_BLA_Antagonist_viability","TOX21_ERa_LUC_BG1_Agonist","TOX21_ERa_LUC_BG1_Antagonist","TOX21_ESRE_BLA_ch1","TOX21_ESRE_BLA_ch2","TOX21_ESRE_BLA_ratio","TOX21_ESRE_BLA_viability","TOX21_FXR_BLA_Antagonist_ch1","TOX21_FXR_BLA_Antagonist_ch2","TOX21_FXR_BLA_agonist_ch2","TOX21_FXR_BLA_agonist_ratio","TOX21_FXR_BLA_antagonist_ratio","TOX21_FXR_BLA_antagonist_viability","TOX21_GR_BLA_Agonist_ch1","TOX21_GR_BLA_Agonist_ch2","TOX21_GR_BLA_Agonist_ratio","TOX21_GR_BLA_Antagonist_ch2","TOX21_GR_BLA_Antagonist_ratio","TOX21_GR_BLA_Antagonist_viability","TOX21_HSE_BLA_agonist_ch1","TOX21_HSE_BLA_agonist_ch2","TOX21_HSE_BLA_agonist_ratio","TOX21_HSE_BLA_agonist_viability","TOX21_MMP_ratio_down","TOX21_MMP_ratio_up","TOX21_MMP_viability","TOX21_NFkB_BLA_agonist_ch1","TOX21_NFkB_BLA_agonist_ch2","TOX21_NFkB_BLA_agonist_ratio","TOX21_NFkB_BLA_agonist_viability","TOX21_PPARd_BLA_Agonist_viability","TOX21_PPARd_BLA_Antagonist_ch1","TOX21_PPARd_BLA_agonist_ch1","TOX21_PPARd_BLA_agonist_ch2","TOX21_PPARd_BLA_agonist_ratio","TOX21_PPARd_BLA_antagonist_ratio","TOX21_PPARd_BLA_antagonist_viability","TOX21_PPARg_BLA_Agonist_ch1","TOX21_PPARg_BLA_Agonist_ch2","TOX21_PPARg_BLA_Agonist_ratio","TOX21_PPARg_BLA_Antagonist_ch1","TOX21_PPARg_BLA_antagonist_ratio","TOX21_PPARg_BLA_antagonist_viability","TOX21_TR_LUC_GH3_Agonist","TOX21_TR_LUC_GH3_Antagonist","TOX21_VDR_BLA_Agonist_viability","TOX21_VDR_BLA_Antagonist_ch1","TOX21_VDR_BLA_agonist_ch2","TOX21_VDR_BLA_agonist_ratio","TOX21_VDR_BLA_antagonist_ratio","TOX21_VDR_BLA_antagonist_viability","TOX21_p53_BLA_p1_ch1","TOX21_p53_BLA_p1_ch2","TOX21_p53_BLA_p1_ratio","TOX21_p53_BLA_p1_viability","TOX21_p53_BLA_p2_ch1","TOX21_p53_BLA_p2_ch2","TOX21_p53_BLA_p2_ratio","TOX21_p53_BLA_p2_viability","TOX21_p53_BLA_p3_ch1","TOX21_p53_BLA_p3_ch2","TOX21_p53_BLA_p3_ratio","TOX21_p53_BLA_p3_viability","TOX21_p53_BLA_p4_ch1","TOX21_p53_BLA_p4_ch2","TOX21_p53_BLA_p4_ratio","TOX21_p53_BLA_p4_viability","TOX21_p53_BLA_p5_ch1","TOX21_p53_BLA_p5_ch2","TOX21_p53_BLA_p5_ratio","TOX21_p53_BLA_p5_viability","Tanguay_ZF_120hpf_AXIS_up","Tanguay_ZF_120hpf_ActivityScore","Tanguay_ZF_120hpf_BRAI_up","Tanguay_ZF_120hpf_CFIN_up","Tanguay_ZF_120hpf_CIRC_up","Tanguay_ZF_120hpf_EYE_up","Tanguay_ZF_120hpf_JAW_up","Tanguay_ZF_120hpf_MORT_up","Tanguay_ZF_120hpf_OTIC_up","Tanguay_ZF_120hpf_PE_up","Tanguay_ZF_120hpf_PFIN_up","Tanguay_ZF_120hpf_PIG_up","Tanguay_ZF_120hpf_SNOU_up","Tanguay_ZF_120hpf_SOMI_up","Tanguay_ZF_120hpf_SWIM_up","Tanguay_ZF_120hpf_TRUN_up","Tanguay_ZF_120hpf_TR_up","Tanguay_ZF_120hpf_YSE_up"],
        'qm8': ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0",
                "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"]
    }

    if dataset_key in label_mapping:
        labels = filtered_data[label_mapping[dataset_key]].values
    elif dataset_key in multi_label_mapping:
        labels = filtered_data[multi_label_mapping[dataset_key]].values
    else:
        raise ValueError(f"No label mapping for '{dataset_name}'")

    if labels.ndim == 1:
        return torch.tensor(labels, dtype=torch.float).unsqueeze(1)
    return torch.tensor(labels, dtype=torch.float)


def compute_pos_weight(labels, train_idx, device):
    """Per-task positive class weight (neg/pos, clipped) for imbalanced BCE loss."""
    y_tr = labels[train_idx].numpy()
    pos = np.nansum(y_tr == 1, axis=0)
    neg = np.nansum(y_tr == 0, axis=0)
    weights = (neg / np.maximum(pos, 1.0)).astype(np.float32)
    weights = np.clip(weights, 1.0, 1e3)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device, task_type, dataset_name=None, scaler=None):
    """Evaluate on a loader; returns ROC-AUC (classification) or RMSE (regression)."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        labels = batch.y
        logits = model(batch)

        if logits.dim() == 2 and labels.dim() == 1:
            batch_size, num_tasks = logits.shape
            labels = labels.view(batch_size, num_tasks)

        if task_type == "classification":
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
        else:
            all_preds.append(logits.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    if task_type == "classification":
        scores = []
        num_tasks = labels.shape[1] if labels.ndim > 1 else 1
        for t in range(num_tasks):
            y_true = labels[:, t] if labels.ndim > 1 else labels
            y_pred = preds[:, t] if preds.ndim > 1 else preds
            mask = ~np.isnan(y_true)
            if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
                continue
            scores.append(roc_auc_score(y_true[mask], y_pred[mask]))
        return float(np.mean(scores)) if scores else 0.0
    else:
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)

        valid = ~np.isnan(labels.flatten())
        preds_flat = preds.flatten()[valid]
        labels_flat = labels.flatten()[valid]
        if dataset_name == "qm8":
            return mean_absolute_error(labels_flat, preds_flat)
        return np.sqrt(mean_squared_error(labels_flat, preds_flat))


def train_epoch(model, loader, optimizer, criterion, device, task_type, scheduler=None, debug=False):
    """Run one fine-tuning epoch; returns the average loss."""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        labels = batch.y.float()

        optimizer.zero_grad()
        logits = model(batch)

        if logits.dim() == 2 and labels.dim() == 1:
            batch_size, num_tasks = logits.shape
            labels = labels.view(batch_size, num_tasks)
        elif logits.dim() == 1 and labels.dim() == 2:
            logits = logits.unsqueeze(1)

        if task_type == "classification":
            mask = ~torch.isnan(labels)
            if mask.sum() == 0:
                continue
            labels_clean = torch.nan_to_num(labels, nan=0.0)
            loss_mat = criterion(logits, labels_clean)
            loss = (loss_mat * mask).sum() / mask.sum()
        else:
            mask = ~torch.isnan(labels)
            if mask.sum() == 0:
                continue
            loss = criterion(logits[mask], labels[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if scheduler is not None and hasattr(scheduler, 'lr'):
            scheduler.step()

    return total_loss / len(loader)


def run_single_run(model, train_loader, valid_loader, test_loader, optimizer, scheduler,
                   criterion, device, task_type, dataset_name, epochs, patience, scaler=None):
    """Train one seed end-to-end with early stopping; returns the best test score."""
    mode = "max" if task_type == "classification" else "min"
    metric_name = "AUC" if task_type == "classification" else ("MAE" if dataset_name == "qm8" else "RMSE")

    best_val = -np.inf if mode == "max" else np.inf
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, task_type, scheduler)
        val_score = evaluate(model, valid_loader, device, task_type, dataset_name, scaler)

        if hasattr(scheduler, 'lr'):
            current_lr = scheduler.lr[0]
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        improved = (val_score > best_val) if mode == "max" else (val_score < best_val)
        if improved:
            best_val = val_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            test_score_tmp = evaluate(model, test_loader, device, task_type, dataset_name, scaler)
            print(f"  Epoch {epoch:03d} | Loss {train_loss:.4f} | Val {metric_name} {val_score:.4f} | Test {test_score_tmp:.4f} | Best {best_val:.4f} | LR {current_lr:.2e}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_score = evaluate(model, test_loader, device, task_type, dataset_name, scaler)
    return test_score, best_epoch, best_state


def main():
    """Fine-tune the pre-trained MAGNET encoder on one benchmark over multiple seeds; report mean +/- std."""
    args = get_conf()

    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--save-checkpoint-dir', type=str, default='', help='Directory to save best checkpoint')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--seed-list', type=str, default='', help='Comma-separated seed list (e.g., 0,42,128)')
    extra_args, _ = parser.parse_known_args()
    save_checkpoint_dir = extra_args.save_checkpoint_dir
    patience = extra_args.patience
    seed_list = [int(s) for s in extra_args.seed_list.split(',')] if extra_args.seed_list else None

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    REGRESSION_DATASETS = ['esol', 'freesolv', 'lipo', 'qm8']
    if args.dataset_name.lower() in REGRESSION_DATASETS:
        args.task_type = 'regression'
    else:
        args.task_type = 'classification'

    smiles_dim = SMILES_DIM_MAP.get(args.smiles_feature_type, 0)

    print(f"Device: {device}")
    print(f"SMILES Feature Type: {args.smiles_feature_type} ({smiles_dim}D)")

    print(f"\nLoading data from: {args.finetuning_graph_pkl}")
    all_data = load_pickle(args.finetuning_graph_pkl)
    dataset_prefix = args.dataset_name.lower()
    filtered_data = all_data[f"{dataset_prefix}_filtered_data"]
    graphs = all_data[f"{dataset_prefix}_all_graphs"]
    print(f"Dataset: {args.dataset_name}, Graphs: {len(graphs)}")

    detected_node_dim = graphs[0].x.shape[1]
    if args.node_dim != detected_node_dim:
        print(f"Auto-detected node_dim: {detected_node_dim}D (overriding --node-dim {args.node_dim})")
        args.node_dim = detected_node_dim
    else:
        print(f"Node Dimension: {args.node_dim}D")

    embedder = get_embedder(args.smiles_feature_type, args.chemberta_model_name, device)
    smiles_dim = embedder.embedding_dim
    print(f"Updated SMILES dimension (dynamic): {smiles_dim}D")

    smiles_list = filtered_data['smiles'].tolist()
    graphs = add_embeddings_to_graphs(graphs, smiles_list, embedder, args.chemberta_batch_size)
    embedder.cleanup()
    del embedder
    torch.cuda.empty_cache()

    labels = get_labels_by_dataset(args.dataset_name, filtered_data)
    num_tasks = labels.shape[1] if labels.ndim > 1 else 1

    if args.pretrained_path:
        pretrained_path = Path(args.pretrained_path)
    else:
        pretrained_path = Path(args.base_dir) / "pretrain_model" / "pretrained_gps.pt"

    train_idx, valid_idx, test_idx = load_split_indices(
        dataset_prefix, args.split_save_dir, args.split_type, args.seeds
    )

    scaler = None
    if args.task_type == "regression" and args.normalize_regression:
        print("Applying StandardScaler to regression targets (fit on train set)")
        scaler = StandardScaler()
        train_labels = labels[train_idx].numpy()
        scaler.fit(train_labels)
        labels_scaled = scaler.transform(labels.numpy())
        labels = torch.tensor(labels_scaled, dtype=torch.float)

    for i, g in enumerate(graphs):
        g.y = labels[i].unsqueeze(0) if labels[i].dim() == 1 else labels[i]
    print(f"Task type: {args.task_type}, Num tasks: {num_tasks}")

    train_graphs = [graphs[i] for i in train_idx]
    valid_graphs = [graphs[i] for i in valid_idx]
    test_graphs = [graphs[i] for i in test_idx]
    print(f"Split: train={len(train_graphs)}, valid={len(valid_graphs)}, test={len(test_graphs)}")

    pos_weight = None
    if args.task_type == "classification":
        pos_weight = compute_pos_weight(labels, train_idx, device)

    metric_name = "AUC" if args.task_type == "classification" else ("MAE" if args.dataset_name == "qm8" else "RMSE")
    all_scores = []
    all_best_epochs = []

    ckpt_dir = None
    if save_checkpoint_dir:
        ckpt_dir = Path(save_checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    effective_proj_dim = getattr(args, 'smiles_proj_dim', 0)

    print(f"\n{'='*60}")
    print(f"Finetuning on {args.dataset_name} ({args.task_type})")
    print(f"SMILES Feature: {args.smiles_feature_type} ({smiles_dim}D)")
    if effective_proj_dim > 0:
        print(f"SMILES Projection: {smiles_dim}D → {effective_proj_dim}D")
        print(f"Combined Dim: {args.model_dim} + {effective_proj_dim} = {args.model_dim + effective_proj_dim}D")
    else:
        print(f"SMILES Projection: None")
        print(f"Combined Dim: {args.model_dim} + {smiles_dim} = {args.model_dim + smiles_dim}D")
    print(f"{'='*60}")

    num_runs = len(seed_list) if seed_list else args.runs
    for run in range(num_runs):
        run_seed = seed_list[run] if seed_list else args.seeds + run
        print(f"\n--- Run {run+1}/{num_runs} (seed={run_seed}) ---")
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        effective_smiles_proj_dim = getattr(args, 'smiles_proj_dim', 0)
        num_layers = args.num_layers
        dropout = args.dropout
        attn_dropout = args.attn_dropout
        batch_size = args.finetuning_batch_size if args.finetuning_batch_size > 0 else 64
        weight_decay = args.finetuning_weight_decay
        warmup_epochs = 2
        if args.task_type == "classification":
            max_lr = args.finetuning_lr
            init_lr = args.finetuning_lr * 0.1
            final_lr = args.finetuning_lr * 0.1
        else:
            max_lr = args.finetuning_lr * 0.1
            init_lr = args.finetuning_lr * 0.01
            if args.dataset_name.lower() == 'lipo':
                final_lr = args.finetuning_lr * 0.001
            else:
                final_lr = args.finetuning_lr * 0.01

        model = GPSForFinetuning(
            node_dim=args.node_dim, hidden_dim=args.model_dim, smiles_dim=smiles_dim,
            num_layers=num_layers, num_heads=args.num_heads,
            dropout=dropout, attn_dropout=attn_dropout,
            pe_dim=args.pe_dim, use_lap_pe=args.use_lap_pe,
            use_degree=args.use_degree, use_rwse=args.use_rwse,
            edge_dim=args.edge_dim, local_gnn_type=args.local_gnn_type,
            output_dim=num_tasks, task_type=args.task_type,
            smiles_proj_dim=effective_smiles_proj_dim
        ).to(device)

        if args.skip_pretrain:
            print("Skipping pretrained weights (random initialization)")
        elif pretrained_path.exists():
            model.load_pretrained_gps(pretrained_path, device)
        else:
            print(f"Warning: Pretrained model not found at {pretrained_path}")

        g = torch.Generator().manual_seed(run_seed)
        train_loader = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True, generator=g)
        valid_loader = PyGDataLoader(valid_graphs, batch_size=batch_size, shuffle=False)
        test_loader = PyGDataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        if args.task_type == "classification":
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        else:
            criterion = nn.HuberLoss(delta=1.0)

        optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)

        scheduler = NoamLR(
            optimizer,
            warmup_epochs=[warmup_epochs],
            total_epochs=[args.finetuning_epochs],
            steps_per_epoch=len(train_loader),
            init_lr=[init_lr], max_lr=[max_lr], final_lr=[final_lr]
        )

        test_score, best_epoch, best_state = run_single_run(
            model, train_loader, valid_loader, test_loader,
            optimizer, scheduler, criterion, device,
            args.task_type, args.dataset_name, args.finetuning_epochs, patience, scaler
        )

        all_scores.append(test_score)
        all_best_epochs.append(best_epoch)
        print(f"  Test {metric_name}: {test_score:.4f} (best epoch: {best_epoch})")

        if ckpt_dir is not None and best_state is not None:
            ckpt_path = ckpt_dir / f"{args.dataset_name}_run{run+1}.pt"
            torch.save({
                'model_state_dict': best_state,
                'dataset': args.dataset_name,
                'test_score': test_score,
                'run': run + 1,
                'best_epoch': best_epoch,
                'seed': run_seed,
                'args': vars(args),
            }, ckpt_path)
            print(f"  Saved run {run+1} checkpoint (epoch {best_epoch}, {metric_name} {test_score:.4f}) -> {ckpt_path}")

    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.dataset_name}")
    print(f"SMILES Feature: {args.smiles_feature_type}")
    print(f"{'='*60}")
    print(f"Test {metric_name}: {mean_score:.4f} +/- {std_score:.4f}")
    print(f"All runs: {[f'{s:.4f}' for s in all_scores]}")
    print(f"{'='*60}")

    if ckpt_dir is not None:
        print(f"Saved {len(all_scores)} run checkpoints to {ckpt_dir} ({args.dataset_name}_run1.pt ... _run{len(all_scores)}.pt)")


if __name__ == '__main__':
    main()

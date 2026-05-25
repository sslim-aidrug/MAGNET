"""Argument parsing and path configuration for MAGNET (pre-training and fine-tuning)."""
import argparse
import os
from pathlib import Path

DEFAULT_BASE_DIR = os.environ.get("MAGNET_BASE_DIR", str(Path(__file__).resolve().parent.parent))

def parse_arguments():
    """Define and parse all command-line arguments for MAGNET."""
    parser = argparse.ArgumentParser(description='GraphGPS for Fragment-level Molecular Graphs')

    # General
    parser.add_argument('--dataset-name', type=str, default='BBBP')
    parser.add_argument('--base-dir', type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--csv-path', type=str, default='', help='Custom CSV path for graph construction')
    parser.add_argument('--graph-pkl', type=str, default='', help='Custom output pickle path')

    # Model architecture (GPS encoder)
    parser.add_argument('--node-dim', type=int, default=933, help='Input node feature dimension (549D RDKit + 384D ChemBERTa)')
    parser.add_argument('--edge-dim', type=int, default=1, help='Edge feature dimension')
    parser.add_argument('--model-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GPS layers')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--attn-dropout', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--local-gnn-type', type=str, default='GATv2', choices=['GINE', 'GATv2'],
                        help='Local MPNN type')

    # Positional / structural encoding
    parser.add_argument('--pe-dim', type=int, default=8, help='Laplacian PE dimension')
    parser.add_argument('--use-lap-pe', type=int, default=1, help='Use Laplacian PE')
    parser.add_argument('--use-degree', type=int, default=1, help='Use Degree encoding')
    parser.add_argument('--use-rwse', type=int, default=0, help='Use RWSE')
    parser.add_argument('--rwse-dim', type=int, default=16, help='RWSE dimension')

    # Pre-training
    parser.add_argument('--pre-lr', type=float, default=1e-4, help='Pretraining learning rate')
    parser.add_argument('--pre-batch-size', type=int, default=256, help='Pretraining batch size')
    parser.add_argument('--pre-epochs', type=int, default=300, help='Pretraining epochs (for Optuna search)')
    parser.add_argument('--final-pre-epochs', type=int, default=100, help='Final pretraining epochs (after Optuna, with best params)')
    parser.add_argument('--pre-graph-pkl', type=str, default='', help='Custom pretraining graph pkl path')
    parser.add_argument('--temperature', type=float, default=0.15, help='Contrastive loss temperature')
    parser.add_argument('--node-mask-ratio', type=float, default=0.2, help='Node masking ratio')
    parser.add_argument('--pre-patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--pre-min-delta', type=float, default=1e-5, help='Early stopping min delta')
    parser.add_argument('--fg-weight', type=float, default=1.0, help='Weight for FG prediction loss in multi-task pretraining')
    parser.add_argument('--property-weight', type=float, default=0.3, help='Weight for property prediction loss in pretraining')
    parser.add_argument('--smiles-graph-weight', type=float, default=0.3, help='Weight for SMILES-Graph contrastive loss in pretraining')
    parser.add_argument('--proj-hidden-dim', type=int, default=256, help='Projection head hidden dimension')
    parser.add_argument('--proj-out-dim', type=int, default=128, help='Projection head output dimension')

    # Fine-tuning
    parser.add_argument('--finetuning-batch-size', type=int, default=0, help='Finetuning batch size (0 = auto: Optuna search or default 64)')
    parser.add_argument('--finetuning-epochs', type=int, default=300, help='Finetuning epochs')
    parser.add_argument('--finetuning-lr', type=float, default=1e-3, help='Finetuning learning rate')
    parser.add_argument('--finetuning-weight-decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--gps-lr-ratio', type=float, default=1.0,
                        help='GPS encoder LR = finetuning_lr * gps_lr_ratio (default: 1.0, same as head)')
    parser.add_argument('--smiles-aug', type=int, default=0,
                        help='Number of random SMILES augmentations per molecule for training (default: 0, no augmentation)')
    parser.add_argument('--chemberta-lr-ratio', type=float, default=0.1,
                        help='ChemBERTa LR = finetuning_lr * ratio (default: 0.1, lower LR for pretrained ChemBERTa)')

    parser.add_argument('--runs', type=int, default=3, help='Number of runs for finetuning')
    parser.add_argument('--skip-pretrain', action='store_true', default=False, help='Skip loading pretrained weights')
    parser.add_argument('--pretrained-path', type=str, default=None, help='Path to pretrained GPS model (.pt file)')

    # SMILES features (concatenated with the graph embedding)
    parser.add_argument('--smiles-feature-type', type=str, default='concat_all',
                        choices=['concat_all'],
                        help='SMILES feature concatenated with the graph embedding (Morgan + ChemBERTa)')
    parser.add_argument('--chemberta-model-name', type=str, default='seyonec/ChemBERTa-zinc-base-v1',
                        help='ChemBERTa model name for transformer-based embeddings')
    parser.add_argument('--chemberta-batch-size', type=int, default=64,
                        help='Batch size for ChemBERTa embedding generation')
    parser.add_argument('--smiles-proj-dim', type=int, default=0,
                        help='Project SMILES embedding to this dimension before concat (0=no projection)')
    parser.add_argument('--normalize-regression', action='store_true', default=True,
                        help='Apply StandardScaler to regression targets (fit on train set only, default: True)')

    parser.add_argument('--aux-weight', type=float, default=0.1,
                        help='Weight for auxiliary task loss (default: 0.1)')

    parser.add_argument('--use-noamlr', action='store_true', default=False,
                        help='Use NoamLR scheduler for all tasks (including classification)')

    # Data split
    parser.add_argument('--seeds', type=int, default=42, help='Random seed')
    parser.add_argument('--split-type', type=str, default='random', help='Split type (random/scaffold)')
    parser.add_argument('--split-size', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Train/Val/Test split ratio (default: 0.8 0.1 0.1)')
    parser.add_argument('--balanced', action='store_true', default=True,
                        help='Use balanced scaffold split')
    parser.add_argument('--no-balanced', action='store_true', default=False,
                        help='Disable balanced scaffold split')
    parser.add_argument('--sort', action='store_true', default=True,
                        help='Sort indices after split')
    parser.add_argument('--stratify', action='store_true', default=True,
                        help='Use stratified split for classification')
    parser.add_argument('--no-stratify', action='store_true', default=False,
                        help='Disable stratified split')
    parser.add_argument('--task-type', type=str, default='classification',
                        choices=['classification', 'regression'], help='Task type')

    parser.add_argument('--use-kfold', action='store_true', default=False,
                        help='Use K-Fold cross validation instead of fixed split')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for K-Fold CV (default: 5)')

    # Data paths
    parser.add_argument('--split-save-dir', type=str, default=None,
                        help='Directory to save split data')

    args, _ = parser.parse_known_args()
    return args


def get_conf():
    """Parse arguments and derive data/checkpoint/pickle paths from base_dir."""
    args = parse_arguments()

    args.base_dir = Path(args.base_dir)

    args.use_lap_pe = bool(args.use_lap_pe)
    args.use_degree = bool(args.use_degree)
    args.use_rwse = bool(args.use_rwse)

    if args.no_stratify:
        args.stratify = False
    if args.no_balanced:
        args.balanced = False

    args.split_size = tuple(args.split_size)

    args.data_dir = args.base_dir / "data"

    args.pre_graph_dir = args.data_dir / "zinc250k_data"
    if args.pre_graph_pkl:
        args.pre_graph_pkl = Path(args.pre_graph_pkl)
    elif args.node_dim == 1317:
        args.pre_graph_pkl = args.pre_graph_dir / "zinc250k_graphs_himol_1317D.pkl"
    elif args.node_dim == 549:
        args.pre_graph_pkl = args.pre_graph_dir / "zinc250k_graphs_gps_enhanced_549d.pkl"
    elif args.node_dim == 2063:
        args.pre_graph_pkl = args.pre_graph_dir / "zinc250k_graphs_gps_morgan.pkl"
    else:
        args.pre_graph_pkl = args.pre_graph_dir / "normalized_zinc250k_graphs_gps_182d.pkl"

    args.graph_dir = args.data_dir / "metagraphs"
    if args.split_save_dir is None:
        args.split_save_dir = args.data_dir / "splits"
    else:
        args.split_save_dir = Path(args.split_save_dir)

    args.graph_dir.mkdir(parents=True, exist_ok=True)

    args.finetuning_graph_pkl = args.graph_dir / f"{args.dataset_name.lower()}_metagraphs.pkl"

    if not args.graph_pkl:
        args.graph_pkl = args.graph_dir / f"{args.dataset_name.lower()}_metagraphs.pkl"

    return args

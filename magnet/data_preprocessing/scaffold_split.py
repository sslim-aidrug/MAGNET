from magnet.data_preprocessing.random_split import *

from collections import defaultdict
import logging
from random import Random
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)
    return scaffolds


def scaffold_split_indices(
    df: pd.DataFrame,
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    balanced: bool = True,
    seed: int = 0,
    sort: bool = True,
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Splits a DataFrame by scaffold so that no molecules sharing a scaffold
    are in different splits. Uses chemprop's original greedy allocation logic.

    :param df: DataFrame with 'smiles' column
    :param sizes: (train, val, test) proportions
    :param balanced: If True, use chemprop's balanced scaffold split
    :param seed: Random seed for balanced splitting
    :param sort: If True, sort returned indices
    :param logger: Optional logger
    :return: (train_idx, val_idx, test_idx) as numpy arrays
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Invalid train/val/test splits! got: {sizes}")

    n = len(df)
    train_size, val_size, test_size = sizes[0] * n, sizes[1] * n, sizes[2] * n
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    smiles_list = df['smiles'].tolist()
    scaffold_to_indices = scaffold_to_smiles(smiles_list, use_indices=True)

    random = Random(seed)

    if balanced:
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')

    print(f'  Scaffolds: total={len(scaffold_to_indices)}, '
          f'train={train_scaffold_count}, val={val_scaffold_count}, test={test_scaffold_count}')

    assert len(set(train) & set(val)) == 0
    assert len(set(test) & set(val)) == 0
    assert len(set(train) & set(test)) == 0

    if sort:
        train = sorted(train)
        val = sorted(val)
        test = sorted(test)

    return np.array(train), np.array(val), np.array(test)


if __name__ == '__main__':

    args = get_conf()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {args.dataset_name}, Seed: {args.seeds}, Device: {device}")

    dataset_name_upper = args.dataset_name

    all_data = load_all_in_one_pickle(args.graph_pkl)

    if all_data is None:
        print("Error: Failed to load pickle file. Exiting.")
        exit()

    dataset_prefix = args.dataset_name.lower()
    data_df_name = f"{dataset_prefix}_filtered_data"
    filtered_data = all_data[data_df_name]

    current_target_list = TARGETS.get(dataset_name_upper)

    if not current_target_list:
        print(f"No TARGETS info for {dataset_name_upper}. Skipping split.")
    else:
        train_idx, val_idx, test_idx = scaffold_split_indices(
            df=filtered_data,
            sizes=args.split_size,
            seed=args.seeds,
            balanced=args.balanced,
            sort=args.sort,)

        print_split_info(
            args.dataset_name,
            filtered_data,
            train_idx,
            val_idx,
            test_idx,
            target_list=current_target_list,
            task_type=args.task_type)

        check_disjoint(train_idx, val_idx, test_idx, len(filtered_data), args.dataset_name)

        save_split_indices(
            args.dataset_name.lower(),
            train_idx,
            val_idx,
            test_idx,
            args.seeds,
            args.split_type,
            dir=args.split_save_dir
        )

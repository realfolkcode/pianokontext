import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import DataLoader

from warping.utils import load_json, prepare_filepaths_from_metadata
from warping.data import EmbeddingDataset


def calculate_dataset_stats(dataset: EmbeddingDataset,
                            batch_size: int) -> Dict[str, List[float]]:
    """Calculates the mean and std stats of a dataset.
    
    Args:
        dataset: Embedding dataset.
        batch_size: Batch size to calculate statistics.
        
    Returns:
        A dictionary with keys `mean` and `std`.
    """
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=8,
                             drop_last=False)
    
    for batch in tqdm(data_loader):
        x_std, x_mean = torch.std_mean(batch, dim=[0, 1])
        break

    data_stats = {"mean": x_mean.tolist(),
                  "std": x_std.tolist()}
    return data_stats


def main(args):
    emb_root_dir = args.data_dir
    metadata_path = args.metadata_path
    seq_len = args.seq_len
    batch_size = args.batch_size
    output_dir = args.output_dir

    metadata = load_json(metadata_path)
    train_path_lst = prepare_filepaths_from_metadata(metadata,
                                                     emb_root_dir=emb_root_dir,
                                                     split='train')
    val_path_lst = prepare_filepaths_from_metadata(metadata,
                                                   emb_root_dir=emb_root_dir,
                                                   split='validation')
    test_path_lst = prepare_filepaths_from_metadata(metadata,
                                                    emb_root_dir=emb_root_dir,
                                                    split='test')

    train_dataset = EmbeddingDataset(emb_path_lst=train_path_lst,
                                     is_cache=False,
                                     seq_len=seq_len)
    val_dataset = EmbeddingDataset(emb_path_lst=val_path_lst,
                                   is_cache=False,
                                   seq_len=seq_len)
    test_dataset = EmbeddingDataset(emb_path_lst=test_path_lst,
                                    is_cache=False,
                                    seq_len=seq_len)
    
    train_stats = calculate_dataset_stats(dataset=train_dataset,
                                          batch_size=batch_size)
    val_stats = calculate_dataset_stats(dataset=val_dataset,
                                        batch_size=batch_size)
    test_stats = calculate_dataset_stats(dataset=test_dataset,
                                         batch_size=batch_size)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_stats_path = os.path.join(output_dir, "train_stats.json")
    val_stats_path = os.path.join(output_dir, "val_stats.json")
    test_stats_path = os.path.join(output_dir, "test_stats.json")

    with open(train_stats_path, "w", encoding="utf8") as f:
        f.write(json.dumps(train_stats, indent=2))

    with open(val_stats_path, "w", encoding="utf8") as f:
        f.write(json.dumps(val_stats, indent=2))

    with open(test_stats_path, "w", encoding="utf8") as f:
        f.write(json.dumps(test_stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='directory with embeddings')
    parser.add_argument('--metadata_path', type=str, required=True, help='path to audio dataset metadata')
    parser.add_argument('--seq_len', type=int, required=True, help='sequence length')
    parser.add_argument('--batch_size', type=int, required=True, help='batch_size')
    parser.add_argument('--output_dir', type=str, required=True, help='directory to store statistics')
    
    args = parser.parse_args()

    main(args)

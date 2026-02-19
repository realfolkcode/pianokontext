import numpy as np
import torch
from dtaidistance import dtw_ndim, ed
from typing import Dict


def cut_embedding(emb,
                  start_s: float,
                  end_s: float,
                  compression_rate = 4096):
    """Cuts an embedding given timestamps and the model compression rate.

    Args:
        emb: Embedding of shape (N, D).
        start_s: The starting timestamp in seconds.
        end: The ending timestemp in seconds.
        compression_rate: The compression rate of audio latent model.
    
    Returns:
        Truncated embedding.
    """
    if not np.isnan(end_s):
        end_idx = int(end_s * 44100 / compression_rate)
        emb = emb[:end_idx]
    
    if not np.isnan(start_s):
        start_idx = int(start_s * 44100 / compression_rate)
        emb = emb[start_idx:]
    
    return emb


def calculate_dtw_path(deadpan_emb: torch.Tensor,
                       expressive_emb: torch.Tensor,
                       window: int = 768) -> Dict:
    """Calculates the DTW path between the embeddings.

    Args:
        deadpan_emb: Deadpan embedding of shape (N, D).
        expressive_emb: Expressive embedding of shape (N, D).
        window: Sakoe-Chiba band in frames.
    
    Returns:
        A dictionary with keys "deadpan" and "expressive" containing
        the corresponding embedding indices.
    """
    deadpan_emb = deadpan_emb.numpy()
    deadpan_emb = deadpan_emb.astype(np.double)

    expressive_emb = expressive_emb.numpy()
    expressive_emb = expressive_emb.astype(np.double)

    max_dist = ed.distance(deadpan_emb,
                           expressive_emb,
                           use_ndim=True)

    dtw_path = dtw_ndim.warping_path(deadpan_emb,
                                     expressive_emb,
                                     window=window,
                                     max_dist=max_dist)
    dtw_path = torch.tensor(dtw_path, dtype=int)

    dtw_path_dict = {"deadpan": dtw_path[:, 0],
                     "expressive": dtw_path[:, 1]}

    return dtw_path_dict

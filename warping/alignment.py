import numpy as np


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

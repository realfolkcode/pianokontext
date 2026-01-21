import os
import torch
from torch.utils.data import Dataset
from typing import List
from copy import deepcopy


class EmbeddingDataset(Dataset):
    def __init__(self,
                 emb_path_lst: List[str],
                 is_cache: bool = True,
                 seq_len: int = 0):
        """Initializes an instance of EmbeddingDataset.

        Args:
            emb_path_lst: A list of file paths.
            is_cache: If True, caches the dataset.
            seq_len: The sequence length to sample. If 0, preserves the whole
              sequence.
        """
        super().__init__()
        self.emb_path_lst = emb_path_lst
        self.is_cache = is_cache
        self.seq_len = seq_len

        self.cached_samples = {}
    
    def __len__(self):
        return len(self.emb_path_lst)

    def __getitem__(self, idx):
        if self.is_cache and idx in self.cached_samples:
            sample = self.cached_samples[idx]
        else:    
            filepath = self.emb_path_lst[idx]
            sample = self._load_sample(filepath)
        
        if self.is_cache and idx not in self.cached_samples:
            self.cached_samples[idx] = sample
        
        if self.seq_len > 0:
            sample = self._sample_subseq(sample)
        
        return sample

    def _load_sample(self,
                     filepath: str) -> torch.Tensor:
        if not os.path.exists(filepath):
            return None
        
        sample = torch.load(filepath).to(torch.float32)
        sample = sample.squeeze().T
        return sample

    def _sample_subseq(self,
                       sample: torch.Tensor) -> torch.Tensor:
        """Samples a subsequence from a sample.

        Args:
            sample: Embedding of shape (N, D).
        
        Returns:
            Embedding of shape (seq_len, D).
        """
        max_start_pos = len(sample) - self.seq_len
        start_pos = torch.randint(low=0, high=max_start_pos + 1, size=(1,))
        end_pos = start_pos + self.seq_len

        new_sample = deepcopy(sample)
        return new_sample[start_pos:end_pos]

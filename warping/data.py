import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from copy import deepcopy


def process_sample_music2latent(sample: torch.Tensor) -> torch.Tensor:
    """Reshapes the sample according to Music2Latent.
    
    Args:
        sample: A Music2Latent sample of shape (1, D, N).
    
    Returns:
        A sample of shape (N, D).
    """
    sample = sample.squeeze().T
    return sample


def process_sample_codicodec(sample: torch.Tensor) -> torch.Tensor:
    """Reshapes the sample according to CoDiCodec.
    
    Args:
        sample: A CoDiCodec sample of shape (N, L, D), where 
          N is the number of time chunks,
          L is the number of latents, and
          D is the dimension of latents.
    
    Returns:
        A sample of shape (N, L * D).
    """
    N = sample.shape[0]
    sample = sample.reshape(N, -1)
    return sample


class EmbeddingDataset(Dataset):
    def __init__(self,
                 emb_dict_lst: List[Dict],
                 is_cache: bool = True,
                 seq_len: int = 0,
                 backbone: str = "music2latent",
                 label_mapping: Dict[str, int] | None = None):
        """Initializes an instance of EmbeddingDataset.

        Args:
            emb_dict_lst: A list of dictionaries with filepaths and meta information.
            is_cache: If True, caches the dataset.
            seq_len: The sequence length to sample. If 0, preserves the whole
              sequence.
            backbone: The backbone of embeddings ("music2latent" or "codicodec").
            label_mapping: Optional mapping from label names to label indices.
        """
        super().__init__()
        self.emb_dict_lst = emb_dict_lst
        self.is_cache = is_cache
        self.seq_len = seq_len
        self.backbone = backbone
        self.label_mapping = label_mapping

        self.cached_samples = {}
        self.cached_labels = {}

        assert self.backbone in ["music2latent", "codicodec"]
    
    def __len__(self):
        return len(self.emb_dict_lst)

    def __getitem__(self, idx):
        if self.is_cache and idx in self.cached_samples:
            sample = self.cached_samples[idx]
            label = self.cached_labels[idx]
        else:    
            filepath = self.emb_dict_lst[idx]["filepath"]
            sample = self._load_sample(filepath)
            if self.label_mapping is not None:
                label = self.emb_dict_lst[idx]["source"]
                label = self.label_mapping[label]
            else:
                label = None
        
        if self.is_cache and idx not in self.cached_samples:
            self.cached_samples[idx] = sample
            self.cached_labels[idx] = label
        
        if self.seq_len > 0:
            sample = self._sample_subseq(sample)
        if self.backbone == "codicodec":
            D = sample.shape[-1]
            sample = sample.reshape(-1, D)
        
        if label is not None:
            return sample, label
        return sample

    def _load_sample(self,
                     filepath: str) -> torch.Tensor:
        """Loads an embedding sample from a filepath.
        
        Args:
            filepath: The path to an embedding file.

        Returns:
            A sample of shape (N, D).
        """
        if not os.path.exists(filepath):
            return None
        
        sample = torch.load(filepath).to(torch.float32)
        
        if self.backbone == "music2latent":
            sample = process_sample_music2latent(sample)
        #elif self.backbone == "codicodec":
        #    sample = process_sample_codicodec(sample)

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

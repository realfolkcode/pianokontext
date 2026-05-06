import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from copy import deepcopy

from .alignment import cut_embedding


class EmbeddingDataset(Dataset):
    def __init__(self,
                 emb_dict_lst: List[Dict],
                 is_cache: bool = True,
                 seq_len: int = 0,
                 label_mapping: Dict[str, int] | None = None):
        """Initializes an instance of EmbeddingDataset.

        Args:
            emb_dict_lst: A list of dictionaries with filepaths and meta information.
            is_cache: If True, caches the dataset.
            seq_len: The sequence length to sample. If 0, preserves the whole
              sequence.
            label_mapping: Optional mapping from label names to label indices.
        """
        super().__init__()
        self.emb_dict_lst = emb_dict_lst
        self.is_cache = is_cache
        self.seq_len = seq_len
        self.label_mapping = label_mapping

        self.cached_samples = {}
        self.cached_labels = {}
    
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
        
        if label is not None:
            return sample, label
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


class AlignedDataset(Dataset):
    def __init__(self,
                 emb_dict_lst: List[Dict],
                 is_cache: bool = True,
                 seq_len: int = 0):
        """Initializes an instance of AlignedDataset.

        Args:
            emb_dict_lst: A list of dictionaries with aligned filepaths.
            is_cache: If True, caches the dataset.
            seq_len: The max sequence length to sample. If 0, preserves the whole
              sequence.
        """
        super().__init__()
        self.emb_dict_lst = emb_dict_lst
        self.is_cache = is_cache
        self.seq_len = seq_len

        self.cached_samples = {}
    
    def __len__(self):
        return len(self.emb_dict_lst)
    
    def _load_expressive_sample(self,
                                emb_dict: Dict) -> torch.Tensor:
        expressive_path = emb_dict["expressive_path"]
        expressive_start_s = emb_dict["expressive_start_s"]
        expressive_end_s = emb_dict["expressive_end_s"]

        expressive_emb = torch.load(expressive_path)
        expressive_emb = expressive_emb.squeeze().T
        expressive_emb = cut_embedding(expressive_emb,
                                       start_s=expressive_start_s,
                                       end_s=expressive_end_s)
        return expressive_emb
    
    def _load_deadpan_sample(self,
                             emb_dict: Dict) -> torch.Tensor:
        deadpan_path = emb_dict["deadpan_path"]
        deadpan_emb = torch.load(deadpan_path)
        deadpan_emb = deadpan_emb.squeeze().T
        return deadpan_emb
    
    def _load_sample(self,
                     emb_dict: Dict) -> Dict:
        deadpan_emb = self._load_deadpan_sample(emb_dict)
        expressive_emb = self._load_expressive_sample(emb_dict)
        
        alignment_filepath = emb_dict["alignment_filepath"]
        dtw_path = torch.load(alignment_filepath)

        sample = {"deadpan": deadpan_emb,
                  "expressive": expressive_emb,
                  "dtw_path": dtw_path}

        if self.seq_len > 0:
            sample["dtw_max_idx"] = self._calculate_max_indices(dtw_path)

        return sample
    
    def __getitem__(self, idx):
        emb_dict = self.emb_dict_lst[idx]

        if idx not in self.cached_samples:
            sample = self._load_sample(emb_dict)
        else:
            sample = self.cached_samples[idx]

        if self.is_cache and idx not in self.cached_samples:
            self.cached_samples[idx] = sample

        if self.seq_len > 0:
            sample = self._sample_subseq(sample)

        return sample

    def _sample_subseq(self,
                       sample: Dict) -> Dict:
        """Samples a subsequence from a sample.
        
        Args:
            sample: A dictionary with deadpan and expressive embeddings,
              DTW path and max indices.
        
        Returns:
            A dictionary with deadpan and expressive subsequences and their
            masks.
        """
        deadpan_emb = sample["deadpan"]
        expressive_emb = sample["expressive"]
        dtw_path = sample["dtw_path"]
        dtw_max_idx = sample["dtw_max_idx"]

        deadpan_idx = dtw_path["deadpan"]
        expressive_idx = dtw_path["expressive"]

        dtw_start = torch.randint(low=0, high=max(len(dtw_max_idx) - 64, 0), size=(1,))
        dtw_end = (dtw_start + dtw_max_idx[dtw_start]) // 2
        dtw_end = torch.randint(low=dtw_end, high=dtw_max_idx[dtw_start] + 1, size=(1,))

        deadpan_start = deadpan_idx[dtw_start]
        deadpan_end = deadpan_idx[dtw_end]
        assert deadpan_end - deadpan_start + 1 <= self.seq_len
        deadpan_emb_sub = torch.zeros((self.seq_len, deadpan_emb.shape[-1]))
        deadpan_emb_sub[:deadpan_end - deadpan_start + 1] = deadpan_emb[deadpan_start:deadpan_end + 1]

        expressive_start = expressive_idx[dtw_start]
        expressive_end = expressive_idx[dtw_end]
        assert expressive_end - expressive_start + 1 <= self.seq_len
        expressive_emb_sub = torch.zeros((self.seq_len, expressive_emb.shape[-1]))
        expressive_emb_sub[:expressive_end - expressive_start + 1] = expressive_emb[expressive_start:expressive_end + 1]

        deadpan_mask = torch.zeros((self.seq_len)).bool()
        deadpan_mask[:deadpan_end - deadpan_start + 1] = 1

        expressive_mask = torch.zeros((self.seq_len)).bool()
        expressive_mask[:expressive_end - expressive_start + 1] = 1

        new_sample = {"deadpan": deadpan_emb_sub,
                      "deadpan_mask": deadpan_mask,
                      "expressive": expressive_emb_sub,
                      "expressive_mask": expressive_mask}
        return new_sample

    def _calculate_max_indices(self,
                               dtw_path: Dict) -> torch.Tensor:
        """
        Calculates the maximum indices for sampling for each starting
        index.

        Args:
            dtw_path: The DTW path indices for deadpan and expressive
              embeddings.
        
        Returns:
            A tensor with maximum indices for sampling for each starting
            index. 
        """
        deadpan_idx = dtw_path["deadpan"]
        expressive_idx = dtw_path["expressive"]

        dtw_max_idx = torch.full_like(deadpan_idx, len(deadpan_idx) - 1)

        start = 0
        end = 0
        while end < len(deadpan_idx):
            deadpan_len = deadpan_idx[end] - deadpan_idx[start] + 1
            expressive_len = expressive_idx[end] - expressive_idx[start] + 1
            if deadpan_len < self.seq_len and expressive_len < self.seq_len:
                end += 1
            else:
                dtw_max_idx[start] = end
                start += 1
        
        assert torch.all(deadpan_idx[dtw_max_idx] >= deadpan_idx)
        assert torch.all(expressive_idx[dtw_max_idx] >= expressive_idx)
        
        assert torch.all(deadpan_idx[dtw_max_idx] - deadpan_idx < self.seq_len)
        assert torch.all(expressive_idx[dtw_max_idx] - expressive_idx < self.seq_len)
        
        return dtw_max_idx

import torch
from torch.func import vmap
from typing import Dict
from functools import partial


## using alpha(t) = (1-t) and beta(t) = t
class DeterministicInterpolant:
    def alpha(self,
              t: float) -> float:
        """Computes alpha coef at time t."""
        return 1.0 - t
    
    def dotalpha(self,
                 t: float) -> float:
        """Computes derivative of alpha wrt time t."""
        return -1.0 + 0*t

    def beta(self,
             t: float) -> float:
        """Computes beta coef at time t."""
        return t
    
    def dotbeta(self,
                t: float) -> float:
        """Computes derivative of beta wrt time t."""
        return 1.0 + 0*t
    
    def _single_xt(self,
                   x0: torch.Tensor,
                   x1: torch.Tensor,
                   t: float) -> torch.Tensor:
        """Interpolates a single pair of samples at time t.
        
        Args:
            x0: Sample at time 0 of shape (seq_len, D).
            x1: Sample at time 1 of shape (seq_len, D).
            t: Time t (scalar).
        
        Returns:
            Interpolated sample of shape (seq_len, D).
        """
        return self.alpha(t) * x0 + self.beta(t) * x1
    
    def _single_dtxt(self,
                     x0: torch.Tensor,
                     x1: torch.Tensor,
                     t: float) -> torch.Tensor:
        """Calculates the vector field for a single pair of samples.

        Args:
            x0: Sample at time 0 of shape (seq_len, D).
            x1: Sample at time 1 of shape (seq_len, D).
            t: Time t (scalar).
        
        Returns:
            The vector field of shape (seq_len, D).
        """
        return self.dotalpha(t) * x0 + self.dotbeta(t) * x1
    
    def xt(self,
           x0: torch.Tensor,
           x1: torch.Tensor,
           t: torch.Tensor) -> torch.Tensor:
        """Interpolates a batch of samples at times t.

        Args:
            x0: Batch of samples at time 0 of shape (B, seq_len, D).
            x1: Batch of samples at time 1 of shape (B, seq_len, D).
            t: Batch of times of shape (B,).
        
        Returns:
            Batch of interpolated samples at times t of shape (B, seq_len, D).
        """
        return vmap(self._single_xt, in_dims=(0, 0, 0))(x0, x1, t)
    
    def dtxt(self,
             x0: torch.Tensor,
             x1: torch.Tensor,
             t: torch.Tensor) -> torch.Tensor:
        """Calculates the vector field for a batch of sample pairs.
        
        Args:
            x0: Batch of samples at time 0 of shape (B, seq_len, D).
            x1: Batch of samples at time 1 of shape (B, seq_len, D).
            t: Batch of times of shape (B,).
        
        Returns:
            Batch of vector fields at times t of shape (B, seq_len, D).
        """
        return vmap(self._single_dtxt, in_dims=(0, 0, 0))(x0, x1, t)

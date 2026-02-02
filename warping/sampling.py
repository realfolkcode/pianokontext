import torch
import torch.nn as nn

from .interpolant import DeterministicInterpolant


@torch.no_grad()
def denoise_latent(model: nn.Module,
                   interpolant: DeterministicInterpolant,
                   seq_len: int,
                   device: str,
                   batch_size: int = 1,
                   dim: int = 64,
                   num_steps: int = 100) -> torch.Tensor:
    """Samples noise and denoises it into a latent code.

    Uses a Heun (midpoint) method to solve ODE.
    
    Args:
        model: Flow matching model.
        interpolant: Flow interpolant.
        seq_len: Sequence length.
        device: Device.
        batch_size: Batch size.
        dim: Latent dimension.
        num_steps: The number of ODE steps.
    
    Returns:
        Latent code of shape (B, seq_len, dim).
    """
    model.eval()

    x = torch.randn((batch_size, seq_len, dim)).to(device)
    t_space = torch.linspace(0, 1, num_steps + 1)[:-1].to(device)
    dt = t_space[1] - t_space[0]
    for t in t_space:
        t = t.unsqueeze(0)
        v = model(x, t)
        v_next = model(x + v * dt, t + dt)
        x = x + (v + v_next) * dt / 2
    
    x = interpolant._unnormalize(x) 
    return x


@torch.no_grad()
def solve_ode(model: nn.Module,
              x_init: torch.Tensor,
              time_grid: torch.Tensor) -> torch.Tensor:
    """Solves an Initial Value Problem with a Flow Matching model.

    Uses a Heun (midpoint) method to solve ODE.
    
    Args:
        model: Flow matching model.
        x_init: Initial condition of shape (B, seq_len, D).
        time_grid: Time grid of shape (num_steps).
    
    Returns:
        Latent code of shape (B, seq_len, dim).
    """
    model.eval()
    x = x_init.clone()
    dt = time_grid[1] - time_grid[0]
    
    for t in time_grid:
        t = t.unsqueeze(0)
        v = model(x, t)
        v_next = model(x + v * dt, t + dt)
        x = x + (v + v_next) * dt / 2
    
    return x


class DualBridge:
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 interpolant_enc: DeterministicInterpolant,
                 interpolant_dec: DeterministicInterpolant):
        """Initializes an instance of DualBridge.
        
        Args:
            encoder: Flow matching model that encodes sample.
            decoder: Flow matching model that decodes sample.
            interpolant_enc: Flow interpolant for encoder.
            interpolant_dec: Flow interpolant for decoder.
        """
        self.encoder = encoder
        self.decoder = decoder

        self.interpolant_enc = interpolant_enc
        self.interpolant_dec = interpolant_dec
    
    def translate_sample(self,
                         x: torch.Tensor,
                         num_steps: int) -> torch.Tensor:
        """Translates sample from the domain of the encoder to the domain
        of the decoder.

        Args:
            x: Sample from the domain of the encoder of shape
              (B, seq_len, D).
            num_steps: The number of denoising steps.
        
        Returns:
            Sample from the domain of the decoder of shape
              (B, seq_len, D).
        """
        t_space = torch.linspace(1, 0, num_steps + 1)[:-1].to(x.device)
        x0 = self.interpolant_enc._normalize(x)
        x0 = solve_ode(model=self.encoder,
                       x_init=x0,
                       time_grid=t_space)
    
        t_space = torch.linspace(0, 1, num_steps + 1)[:-1].to(x.device)
        x1_rec = solve_ode(model=self.decoder,
                           x_init=x0,
                           time_grid=t_space)
        x1_rec = self.interpolant_dec._unnormalize(x1_rec)
        
        return x1_rec

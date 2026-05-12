import torch
import torch.nn as nn

from .interpolant import DeterministicInterpolant


def cfg_step(model: nn.Module,
             x: torch.Tensor,
             t: torch.Tensor,
             y: torch.Tensor,
             cfg_strength: float) -> torch.Tensor:
    """CFG velocity prediction.

    Args:
        model: Flow matching model.
        x: A noisy latent of shape (B, seq_len, dim).
        t: Batch of times of shape (B,).
        y: Labels of shape (B,).        
        cfg_strength: Classifier-free guidance strength.
    
    Returns:
        CFG velocity of shape (B, seq_len, dim).
    """
    num_classes = model.y_embedder.num_classes
    y_null = torch.tensor([num_classes] * len(y), device=y.device)

    v = model(x, t, y)
    v_null = model(x, t, y_null)

    v = (1 - cfg_strength) * v_null + cfg_strength * v
    return v


@torch.no_grad()
def denoise_latent(model: nn.Module,
                   interpolant: DeterministicInterpolant,
                   seq_len: int,
                   device: str,
                   x: torch.Tensor | None = None,
                   y: torch.Tensor | None = None,
                   cfg_strength: float | None = None,
                   batch_size: int | None = 1,
                   dim: int = 64,
                   num_steps: int = 100) -> torch.Tensor:
    """Samples noise and denoises it into a latent code.

    Uses a Heun (midpoint) method to solve ODE.
    
    Args:
        model: Flow matching model.
        interpolant: Flow interpolant.
        seq_len: Sequence length.
        device: Device.
        x: A noisy latent of shape (B, seq_len, dim).
        y: Labels of shape (B,). If passed, applies CFG.
        cfg_strength: Classifier-free guidance strength.
        batch_size: Batch size.
        dim: Latent dimension.
        num_steps: The number of ODE steps.
    
    Returns:
        Latent code of shape (B, seq_len, dim).
    """
    model.eval()

    if x is None:
        x = torch.randn((batch_size, seq_len, dim)).to(device)
    
    t_space = torch.linspace(0, 1, num_steps + 1)[:-1].to(device)

    x = solve_ode(model=model,
                  x_init=x,
                  time_grid=t_space,
                  y=y,
                  cfg_strength=cfg_strength)
    x = interpolant._unnormalize(x) 
    return x


@torch.no_grad()
def solve_ode(model: nn.Module,
              x_init: torch.Tensor,
              time_grid: torch.Tensor,
              y: torch.Tensor | None = None,
              cfg_strength: float | None = None) -> torch.Tensor:
    """Solves an Initial Value Problem with a Flow Matching model.

    Uses a Heun (midpoint) method to solve ODE.
    
    Args:
        model: Flow matching model.
        x_init: Initial condition of shape (B, seq_len, D).
        time_grid: Time grid of shape (num_steps).
        y: Labels of shape (B,). If passed, applies CFG.
        cfg_strength: Classifier-free guidance strength.
    
    Returns:
        Latent code of shape (B, seq_len, dim).
    """
    model.eval()
    x = x_init.clone()
    dt = time_grid[1] - time_grid[0]
    
    for t in time_grid:
        t = t.unsqueeze(0)
        if y is None:
            v = model(x, t)
            v_next = model(x + v * dt, t + dt)
            x = x + (v + v_next) * dt / 2
        else:
            v = cfg_step(model, x, t, y, cfg_strength)
            v_next = cfg_step(model,
                              x + v * dt,
                              t + dt,
                              y,
                              cfg_strength)
            x = x + v * dt
    
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


@torch.no_grad()
def solve_ode_flux(model: nn.Module,
                   x_init: torch.Tensor,
                   x_init_mask: torch.Tensor,
                   time_grid: torch.Tensor,
                   context: torch.Tensor,
                   context_mask: torch.Tensor) -> torch.Tensor:
    """Solves an Initial Value Problem with a Flux-style Flow Matching model.

    Uses a Heun (midpoint) method to solve ODE.
    
    Args:
        model: Flow matching model.
        x_init: Initial condition of shape (B, seq_len, D).
        x_init_mask: Mask for initial condition of shape (B, seq_len).
        time_grid: Time grid of shape (num_steps).
        context: Context of shape (B, seq_len, D).
        context_mask: Mask for context of shape (B, seq_len).
    
    Returns:
        Latent code of shape (B, seq_len - 1, dim).
    """
    model.eval()
    x = x_init.clone()
    dt = time_grid[1] - time_grid[0]
    
    for t in time_grid:
        t = t.unsqueeze(0)
        
        v = model(x=x,
                  x_mask=x_init_mask,
                  context=context,
                  context_mask=context_mask,
                  t=t)
        v_next = model(x=x + v * dt,
                       x_mask=x_init_mask,
                       context=context,
                       context_mask=context_mask,
                       t=t + dt)
        x = x + (v + v_next) * dt / 2
    
    return x

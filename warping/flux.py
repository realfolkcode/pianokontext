import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from typing import Dict

from .attention import MaskedAttention, precompute_freqs_cis
from .sit import modulate, TimeEmbedder, FinalLayer


class FluxBlock(nn.Module):
    """Flux-style block."""

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0):
        """Initilalizes an instance of SiTBlock.
        
        Args:
            hidden_size: The dimension of hidden layers.
            num_heads: The number of heads in attention layers.
            mlp_ratio: The ratio of hidden size in MLP.
        """
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MaskedAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu,
                       drop=0)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                c: torch.Tensor,
                freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the SiTBlock.
        
        Args:
            x: Batch of latents of shape (B, 2 * seq_len, hid_dim).
            mask: Batch of masks of shape (B, 2 * seq_len, seq_len).
            c: Batch of conditioning tensors of shape (B, hid_dim).
            freqs_cis: (Optional) Rotary embeddings of shape
              (2 * seq_len, hid_dim // (2 * num_heads)).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),
                                                  mask,
                                                  freqs_cis)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class Flux(nn.Module):
    """Flux-style diffusion transformer."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_size: int,
                 num_blocks: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 seq_len: int = 128):
        """Initializes an instance of DiT.
        
        Args:
            input_dim: The input dimension.
            hidden_size: The dimension of hidden layers.
            num_layers: The number of SiT blocks.
            num_heads: The number of attention heads.
            mlp_ratio: The ratio of hidden size in MLP.
            seq_len: The sequence length.
        """
        super().__init__()

        self.projection = nn.Linear(input_dim, hidden_size, bias=True)

        self.freqs_cis = precompute_freqs_cis(dim=hidden_size // num_heads,
                                              end=2 * seq_len)
        
        self.t_embedder = TimeEmbedder(out_dim=hidden_size)

        self.blocks = nn.ModuleList([
            FluxBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_blocks)
        ])
        self.final_layer = FinalLayer(hidden_size=hidden_size,
                                      input_dim=input_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self,
                x: torch.Tensor,
                x_mask: torch.Tensor,
                context: torch.Tensor,
                context_mask: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Batch of samples of shape (B, seq_len, D).
            x_mask: Batch of masks for x of shape (B, seq_len).
            context: Batch of context samples of shape (B, seq_len, D).
            context_mask: Batch of masks for c of shape (B, seq_len).
            t: Batch of times of shape (B,).
        
        Returns:
            Batch of vector fields of shape (B, seq_len, D).
        """
        self.freqs_cis = self.freqs_cis.to(x.device)

        context_seq_len = context.shape[1]
        x = torch.concat((context, x), dim=1)   # (B, 2 * seq_len, D)

        mask = torch.concat((context_mask, x_mask), dim = 1)   # (B, 2 * seq_len)
        mask = mask.unsqueeze(-1)               # (B, 2 * seq_len, 1)
        mask = mask * mask.transpose(1, 2)      # (B, 2 * seq_len, 2 * seq_len)
        mask = mask.unsqueeze(1)                # (B, 1, 2 * seq_len, 2 * seq_len)

        x = self.projection(x)                  # (B, 2 * seq_len, hid_dim)
        
        t = self.t_embedder(t)                  # (B, hid_dim)
        
        for i, block in enumerate(self.blocks):
            x = block(x, mask, t, self.freqs_cis)     # (B, 2 * seq_len, hid_dim)

        x = x[:, context_seq_len:]                 # (B, seq_len, D)
        
        x = self.final_layer(x, t)              # (B, seq_len, D)
        return x


def prepare_flux_from_config(config: Dict,
                             device: str) -> nn.Module:
    """Prepares an instance of Flux from a config.

    Args:
        config: A dictionary containing model hyperparameters.
        device: Device.
    
    Returns:
        A Flux model with hyperparameters.
    """
    input_dim = config['data']['input_dim']
    hidden_size = config['model']['hidden_size']
    num_blocks = config['model']['num_blocks']
    num_heads = config['model']['num_heads']
    mlp_ratio = config['model']['mlp_ratio']

    sit = Flux(input_dim=input_dim,
               hidden_size=hidden_size,
               num_blocks=num_blocks,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio).to(device)
    
    return sit

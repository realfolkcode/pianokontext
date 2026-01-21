import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict
#from timm.models.vision_transformer import Attention
from timm.models.vision_transformer import Mlp

from .attention import Attention, precompute_freqs_cis


class TimeEmbedder(nn.Module):
    """Embeds timesteps into vector representations.
    
    References:
    GLIDE: https://github.com/openai/glide-text2im
    SiT: https://github.com/willisma/SiT
    """

    def __init__(self,
                 out_dim: int,
                 freq_emb_dim: int = 256):
        """Initializes an instance of TimeEmbedder.

        Args:
            out_dim: The output dimension of time embeddings.
            freq_emb_dim: The dimension of sinusoidal embeddings.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_emb_dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True)
        )
        self.freq_emb_dim = freq_emb_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, time_factor=1000.):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self,
                t: torch.Tensor) -> torch.Tensor:
        """Forward pass of a time embedder.

        Args:
            t: Batch of times of shape (B,).
        
        Returns:
            Time embeddings of shape (B, out_dim).
        """
        t_freq = self.timestep_embedding(t, self.freq_emb_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    """Rescales and shifts each element of tokens by parameters.

    Args:
        x: Batch of latents of shape (B, seq_len, hid_dim).
        shift: Batch of shifts of shape (B, hid_dim).
        scale: Batch of scales of shape (B, hid_dim).
    
    Returns:
        Modulated latents of shape (B, seq_len, hid_dim).
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SiTBlock(nn.Module):
    """SiT block with adaptive layer norm zero conditioning."""

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
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu,
                       drop=0)

    def forward(self,
                x: torch.Tensor,
                c: torch.Tensor,
                freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the SiTBlock.
        
        Args:
            x: Batch of latents of shape (B, seq_len, hid_dim).
            c: Batch of conditioning tensors of shape (B, hid_dim).
            freqs_cis: (Optional) Rotary embeddings of shape
              (seq_len, hid_dim // (2 * num_heads)).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),
                                                  freqs_cis)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of SiT."""

    def __init__(self,
                 hidden_size: int,
                 input_dim: int):
        """Initializes an instance of FinalLayer.
        
        Args:
            hidden_size: The dimension of hidden layers of SiT.
            input_dim: The input dimension of SiT.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, input_dim, bias=True)

    def forward(self,
                x: torch.Tensor,
                c: torch.Tensor):
        """Forward pass of FinalLayer.
        
        Args:
            x: Batch of latents of shape (B, seq_len, hid_dim).
            c: Batch of conditioning tensors of shape (B, hid_dim).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """Stochastic interpolant transformer for learning vector fields."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_size: int,
                 num_blocks: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0):
        """Initializes an instance of DiT.
        
        Args:
            input_dim: The input dimension.
            hidden_size: The dimension of hidden layers.
            num_layers: The number of SiT blocks.
            num_heads: The number of attention heads.
            mlp_ratio: The ratio of hidden size in MLP.
        """
        super().__init__()

        self.projection = nn.Linear(input_dim, hidden_size, bias=True)

        self.freqs_cis = precompute_freqs_cis(dim=hidden_size // num_heads,
                                              end=128)
        
        self.t_embedder = TimeEmbedder(out_dim=hidden_size)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_blocks)
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
                t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Batch of samples of shape (B, seq_len, D).
            t: Batch of times of shape (B,).
        
        Returns:
            Batch of vector fields of shape (B, seq_len, D).
        """
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = self.projection(x)                  # (B, seq_len, hid_dim)
        
        t = self.t_embedder(t)                  # (B, hid_dim)
        
        for i, block in enumerate(self.blocks):
            x = block(x, t, self.freqs_cis)     # (B, seq_len, hid_dim)
        x = self.final_layer(x, t)              # (B, seq_len, D)

        return x


def prepare_sit_from_config(config: Dict,
                            device: str) -> SiT:
    """Prepares an instance of SiT from a config.

    Args:
        config: A dictionary containing model hyperparameters.
        device: Device.
    
    Returns:
        A SiT model with hyperparameters.
    """
    input_dim = config['data']['input_dim']
    hidden_size = config['model']['hidden_size']
    num_blocks = config['model']['num_blocks']
    num_heads = config['model']['num_heads']
    mlp_ratio = config['model']['mlp_ratio']

    sit = SiT(input_dim=input_dim,
              hidden_size=hidden_size,
              num_blocks=num_blocks,
              num_heads=num_heads,
              mlp_ratio=mlp_ratio).to(device)
    
    return sit

from ast import Mult
from unittest.mock import patch
import torch
import torch.nn as nn

from models.common.layers import MultiHeadAttention, MLPBlock
from utils.image import slice_image_to_patches


class EncoderBlock(nn.Module):
    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_dim)
        self.multi_head_attention = MultiHeadAttention(n_dim=n_dim, n_heads=n_heads)
        self.mlp_block = MLPBlock(n_dim=n_dim)

    def forward(self, x):
        x_backup = x
        x = self.layer_norm(x)
        x = self.multi_head_attention(x)
        x = x_backup = x + x_backup
        x = self.layer_norm(x)
        x = self.mlp_block(x)
        x += x_backup
        return x

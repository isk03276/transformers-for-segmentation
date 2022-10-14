import torch.nn as nn

from transformers_for_segmentation.common.attention.deformable_attention import (
    DeformableAttention,
)
from transformers_for_segmentation.common.layers import MLPBlock


class EncoderBlock(nn.Module):
    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_dim)
        self.deformable_attention = DeformableAttention(n_dim=n_dim, n_heads=n_heads)
        self.mlp_block = MLPBlock(n_dim=n_dim)

    def forward(self, x):
        x_backup = x
        # x = self.layer_norm(x)
        x = self.deformable_attention(x)
        x = x_backup = x + x_backup
        # x = self.layer_norm(x)
        x = self.mlp_block(x)
        x += x_backup
        return x

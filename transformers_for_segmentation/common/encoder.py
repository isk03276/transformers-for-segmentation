import torch.nn as nn

from transformers_for_segmentation.common.layers import MLPBlock
from transformers_for_segmentation.common.attention.multi_head_self_attention import MultiHeadSelfAttention


class EncoderBlock(nn.Module):
    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_dim)
        self.multi_head_attention = MultiHeadSelfAttention(n_dim=n_dim, n_heads=n_heads)
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

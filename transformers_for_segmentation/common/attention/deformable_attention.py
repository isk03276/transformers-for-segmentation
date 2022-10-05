import torch

from transformers_for_segmentation.common.attention.multi_head_self_attention import MultiHeadSelfAttention


class DeformableAttention(MultiHeadSelfAttention):
    def __init__(self, n_dim:int, n_heads: int):
        super().__init__(n_dim=n_dim, n_heads=n_heads)
        
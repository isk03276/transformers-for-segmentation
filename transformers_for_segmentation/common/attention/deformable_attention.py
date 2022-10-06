import torch
import torch.nn as nn

from transformers_for_segmentation.common.attention.multi_head_self_attention import MultiHeadSelfAttention


class DeformableAttention(MultiHeadSelfAttention):
    def __init__(self, n_dim:int, n_heads: int):
        super().__init__(n_dim=n_dim, n_heads=n_heads)
        self.offset_net = OffsetNet(n_dim=n_dim)
        
    def forward(self, x):
        queries = self.split_multiple_heads(self.query(x))
        offset = self.offset_net(queries)
        
        
        
        keys = self.split_multiple_heads(self.key(x))
        values = self.split_multiple_heads(self.value(x))

        result = self.scaled_dot_proudction(query=queries, key=keys, value=values)
        result = self.merge_multiple_heads(result)
        result = self.linear(result)
        return result
        
        
class OffsetNet(nn.Module):
    def __init__(self, n_dim:int, n_offset:int, n_heads:int, kernel_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(in_channels=n_dim, out_channels=n_dim, kernel_size=kernel_size, ),
                                 nn.GELU(),
                                 nn.Conv3d(in_channels=n_dim, out_channels=3, kernel_size=1),
                                 nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)

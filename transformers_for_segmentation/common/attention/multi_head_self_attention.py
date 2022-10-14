import torch
import torch.nn as nn

from transformers_for_segmentation.common.attention.base_attention import BaseAttention


class MultiHeadSelfAttention(BaseAttention):
    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        assert n_dim % n_heads == 0

        self.n_dim = n_dim
        self.n_heads = n_heads
        self.scaling_factor = n_dim ** (-0.5)

        self.to_query = nn.Linear(n_dim, n_dim)
        self.to_key = nn.Linear(n_dim, n_dim)
        self.to_value = nn.Linear(n_dim, n_dim)
        self.out = nn.Linear(n_dim, n_dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)
        attention_map = self.get_attention_map(query=query, key=key)
        result = self.do_attention(attention_map=attention_map, value=value)
        result = self.out(result)
        return result

    def do_attention(self, attention_map: torch.Tensor, value: torch.Tensor):
        value = self.split_multiple_heads(value)
        result = torch.matmul(attention_map, value)
        result = self.merge_multiple_heads(result)
        return result

    def get_attention_map(self, query: torch.Tensor, key: torch.Tensor):
        query = self.split_multiple_heads(query)
        key = self.split_multiple_heads(key)
        # (B, n_head, SeqN, splitted_dim)
        transposed_key = key.transpose(-2, -1)
        attention_map = torch.matmul(query, transposed_key) * self.scaling_factor
        attention_map = self.softmax(attention_map)
        return attention_map

    def split_multiple_heads(self, tensor: torch.Tensor):
        assert len(tensor.size()) == 3

        batch_size, seq_len, emb_dim = tensor.size()
        # (B, SeqN, Emb) -> (B, SeqN, n_head, splitted_dim)
        tensor = tensor.reshape(
            batch_size, seq_len, self.n_heads, emb_dim // self.n_heads
        )
        # (B, SeqN, n_head, splitted_dim) -> (B, n_head, SeqN, splitted_dim)
        tensor = tensor.transpose(1, 2)
        return tensor

    def merge_multiple_heads(self, tensor: torch.Tensor):
        assert len(tensor.size()) == 4

        batch_size, n_heads, seq_len, splitted_dim = tensor.size()
        # (B, n_head, SeqN, splitted_dim) -> (B, SeqN, n_head, splitted_dim)
        tensor = tensor.transpose(1, 2)
        # (B, SeqN, n_head, splitted_dim) -> (B, SeqN, n_head * splitted_dim)
        tensor = tensor.reshape(batch_size, seq_len, n_heads * splitted_dim)
        return tensor

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_dim: int, n_heads: int):
        super().__init__()
        assert n_dim % n_heads == 0

        self.n_dim = n_dim
        self.n_heads = n_heads
        self.scaling_factor = n_dim ** (-0.5)

        self.query = nn.Linear(n_dim, n_dim)
        self.key = nn.Linear(n_dim, n_dim)
        self.value = nn.Linear(n_dim, n_dim)

        self.linear = nn.Linear(n_dim, n_dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        queries = self.split_multiple_heads(self.query(x))
        keys = self.split_multiple_heads(self.key(x))
        values = self.split_multiple_heads(self.value(x))

        result = self.scaled_dot_proudction(query=queries, key=keys, value=values)
        result = self.merge_multiple_heads(result)
        result = self.linear(result)
        return result

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

    def scaled_dot_proudction(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        # (B, n_head, SeqN, splitted_dim)
        transposed_key = key.transpose(-2, -1)
        attention_score_map = torch.matmul(query, transposed_key) * self.scaling_factor
        attention_prob_map = self.softmax(attention_score_map)
        result = torch.matmul(attention_prob_map, value)
        return result

    def merge_multiple_heads(self, tensor: torch.Tensor):
        assert len(tensor.size()) == 4

        batch_size, n_heads, seq_len, splitted_dim = tensor.size()
        # (B, n_head, SeqN, splitted_dim) -> (B, SeqN, n_head, splitted_dim)
        tensor = tensor.transpose(1, 2)
        # (B, SeqN, n_head, splitted_dim) -> (B, SeqN, n_head * splitted_dim)
        tensor = tensor.reshape(batch_size, seq_len, n_heads * splitted_dim)
        return tensor

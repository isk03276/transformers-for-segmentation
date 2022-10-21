import torch
import einops

from transformers_for_segmentation.common.attention.deformable_attention import (
    DeformableAttention,
)


class PatchWiseDeformableAttention(DeformableAttention):
    def __init__(
        self,
        n_dim: int,
        n_heads: int,
        n_groups: int,
        use_dynamic_positional_encoding: str,
        offset_kernel_size: int = 5,
        offset_scale_factor: int = 4,
    ):
        super().__init__(
            n_dim=n_dim,
            n_heads=n_heads,
            n_groups=n_groups,
            use_dynamic_positional_encoding=use_dynamic_positional_encoding,
            offset_kernel_size=offset_kernel_size,
            offset_scale_factor=offset_scale_factor,
        )
        assert n_dim % n_heads == 0, "n_dim % n_heads must be 0."
        assert n_dim % n_groups == 0, "n_dim % n_groups must be 0."

    def do_attention(
        self, attention_map: torch.Tensor, value: torch.Tensor, x_size: tuple,
    ):
        batch_size, _, depth, height, width = x_size
        value = self.group_feature(value)
        value, _ = self.split_heads((value, value))  # to be modified
        value = einops.rearrange(
            value,
            "(batch groups) heads dhw emb -> (batch heads dhw) groups emb",
            batch=batch_size,
        )
        result = torch.bmm(attention_map, value)
        result = einops.rearrange(
            result,
            "(batch heads dhw) groups emb -> batch (heads emb groups) dhw",
            batch=batch_size,
            heads=self.n_heads,
            groups=self.n_groups,
        )
        result = einops.rearrange(
            result, "batch emb (d h w) -> batch emb d h w", d=depth, h=height, w=width
        )
        return result

    def get_attention_map(
        self, query: torch.Tensor, key: torch.Tensor,
    ):
        batch_size, _, _, _, _ = query.size()
        query = self.group_feature(query)
        key = self.group_feature(key)
        query, key = self.split_heads((query, key))

        query = einops.rearrange(
            query,
            "(batch groups) heads dhw emb -> (batch heads dhw) groups emb",
            batch=batch_size,
        )
        key = einops.rearrange(
            key,
            "(batch groups) heads dhw emb -> (batch heads dhw) emb groups",
            batch=batch_size,
        )
        attention_map = torch.bmm(query, key)
        attention_map *= self.scaling_factor
        attention_map = attention_map.softmax(dim=-1)
        return attention_map

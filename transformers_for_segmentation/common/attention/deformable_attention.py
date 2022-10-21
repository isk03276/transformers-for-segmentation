from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformers_for_segmentation.common.attention.base_attention import BaseAttention


class DeformableAttention(BaseAttention):
    """
    Multi-head deforable attention class.
    """

    def __init__(
        self,
        n_dim: int,
        n_heads: int,
        n_groups: int,
        use_dynamic_positional_encoding: str,
        offset_kernel_size: int = 5,
        offset_scale_factor: int = 4,
    ):
        super().__init__()
        assert n_dim % n_heads == 0, "n_dim % n_heads must be 0."
        assert n_dim % n_groups == 0, "n_dim % n_groups must be 0."

        self.n_dim = n_dim
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.use_dynamic_positional_encoding = use_dynamic_positional_encoding
        self.offset_kernel_size = offset_kernel_size
        self.offset_scale_factor = offset_scale_factor
        self.scaling_factor = n_dim ** (-0.5)

        self.n_head_dim = n_dim // n_heads  # nhc
        self.n_group_dim = n_dim // n_groups  # ngc
        self.n_group_heads = n_heads // n_groups  # ngh

        self.offset_net = nn.Sequential(
            nn.Conv3d(
                in_channels=self.n_group_dim,
                out_channels=self.n_group_dim,
                kernel_size=self.offset_kernel_size,
                groups=self.n_groups,
                padding=self.offset_kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv3d(
                in_channels=self.n_group_dim, out_channels=3, kernel_size=1, bias=False
            ),
            nn.Tanh(),
        )

        self.position_bias = nn.Parameter(torch.randn(self.n_dim, 6, 6, 6))

        self.to_query = nn.Conv3d(
            in_channels=n_dim,
            out_channels=n_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.to_key = nn.Conv3d(
            in_channels=n_dim,
            out_channels=n_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.to_value = nn.Conv3d(
            in_channels=n_dim,
            out_channels=n_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.out = nn.Conv3d(in_channels=n_dim, out_channels=n_dim, kernel_size=1)

    def forward(self, x):
        query = self.to_query(x)
        grouped_query = self.group_feature(query)

        deformed_points = self.get_offset(grouped_query=grouped_query)
        x_sampled = self.sample_from_offset(x=x, deformed_points=deformed_points)
        if self.use_dynamic_positional_encoding:
            x_sampled += self.position_bias

        key = self.to_key(x_sampled)
        value = self.to_value(x_sampled)
        attention = self.get_attention_map(query=query, key=key)
        result = self.do_attention(
            attention_map=attention, value=value, x_size=x.size(),
        )
        result = self.out(result)
        return result

    def do_attention(
        self, attention_map: torch.Tensor, value: torch.Tensor, x_size: tuple,
    ):
        _, _, depth, height, width = x_size
        value, _ = self.split_heads((value, value))  # to be modified
        # result = torch.einsum('batch n_head temp_i temp_j, batch n_head temp_j emb -> batch n_head temp_i emb', attention_map, value)
        result = torch.einsum("b h i j, b h j e -> b h i e", attention_map, value)
        result = einops.rearrange(
            result,
            "batch n_head (depth height width) emb -> batch (n_head emb) depth height width",
            depth=depth,
            height=height,
            width=width,
        )
        return result

    def get_attention_map(
        self, query: torch.Tensor, key: torch.Tensor,
    ):
        query, key = self.split_heads((query, key))
        # attention_map = torch.einsum('batch n_head temp_i emb, batch n_head temp_j emb -> batch emb temp_i temp_j', query, key)
        attention_map = torch.einsum("b h i d, b h j d -> b h i j", query, key)
        attention_map *= self.scaling_factor
        attention_map = attention_map.softmax(dim=-1)
        return attention_map

    def get_offset(self, grouped_query: torch.Tensor):
        offset = self.offset_net(grouped_query)
        grid = self.create_grid(grouped_query)
        deformed_points = self.normalize_grid(grid=offset + grid)
        return deformed_points

    def sample_from_offset(self, x: torch.Tensor, deformed_points: torch.Tensor):
        x_sampled = F.grid_sample(
            self.group_feature(x),
            deformed_points,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return self.ungroup_feature(x_sampled)

    def group_feature(self, feature: torch.Tensor):
        assert len(feature.shape) == 5  # batch emb depth height width

        grouped_feature = einops.rearrange(
            feature,
            "batch (n_group n_group_dim) depth height width -> (batch n_group) n_group_dim depth height width",
            n_group=self.n_groups,
        )
        return grouped_feature

    def ungroup_feature(self, feature: torch.Tensor):
        assert len(feature.shape) == 5  # batch emb depth height width

        ungrouped_feature = einops.rearrange(
            feature,
            "(batch n_group) n_group_dim depth height width  -> batch (n_group n_group_dim) depth height width",
            n_group=self.n_groups,
        )
        return ungrouped_feature

    def split_heads(self, features: Tuple[torch.Tensor, ...]):
        result = map(
            lambda t: einops.rearrange(t, "b (h d) ... -> b h (...) d", h=self.n_heads),
            features,
        )
        return result

    def create_grid(self, tensor, dim=0):
        depth, height, width, device = *tensor.shape[2:], tensor.device
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(depth, device=device),
                torch.arange(height, device=device),
                torch.arange(width, device=device),
            ),
            dim=dim,
        )
        grid.requires_grad = False
        grid = grid.type_as(tensor)
        return grid

    def normalize_grid(self, grid, dim=1, out_dim=-1):
        d, h, w = grid.shape[-3:]
        grid_d, grid_h, grid_w = grid.unbind(dim=dim)

        grid_d = 2.0 * grid_d / max(d - 1, 1) - 1.0
        grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
        grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

        return torch.stack((grid_d, grid_h, grid_w), dim=out_dim)

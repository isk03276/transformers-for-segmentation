import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformers_for_segmentation.common.attention.multi_head_self_attention import (
    MultiHeadSelfAttention,
)


class DeformableAttention(MultiHeadSelfAttention):
    def __init__(self, n_dim: int, n_heads: int, n_groups: int, kernel_size: int = 5):
        super().__init__(n_dim=n_dim, n_heads=n_heads)
        assert n_dim % n_groups == 0, "n_dim % n_groups must be 0."

        self.n_head_dim = n_dim // n_heads  # nhc
        self.n_group_dim = n_dim // n_groups  # ngc
        self.n_group_heads = n_heads // n_groups  # ngh

        self.n_groups = n_groups
        self.offset_net = nn.Sequential(
            nn.Conv3d(
                in_channels=self.n_group_dim,
                out_channels=self.n_group_dim,
                kernel_size=kernel_size,
                groups=n_groups,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv3d(
                in_channels=self.n_group_dim, out_channels=3, kernel_size=1, bias=False
            ),
            nn.Tanh(),
        )
        self.query = nn.Conv3d(
            in_channels=n_dim, out_channels=n_dim, kernel_size=1, stride=1, padding=0
        )
        self.key = nn.Conv3d(
            in_channels=n_dim, out_channels=n_dim, kernel_size=1, stride=1, padding=0
        )
        self.value = nn.Conv3d(
            in_channels=n_dim, out_channels=n_dim, kernel_size=1, stride=1, padding=0
        )
        self.out = nn.Conv3d(in_channels=n_dim, out_channels=n_dim, kernel_size=1)

    def forward(self, x):
        batch_size, n_dim, depth, height, width = x.size()
        queries = self.query(x)
        grouped_queries = einops.rearrange(
            queries, "b (g c) d h w  -> (b g) c d h w", g=self.n_groups,
        )
        offset = self.offset_net(grouped_queries)
        grid = self.create_grid_like(x)
        deformed_points = offset + grid
        deformed_points = self.normalize_grid(deformed_points)

        x_sampled = F.grid_sample(
            einops.rearrange(x, "b (g gc) d h w -> (b g) gc d h w", g=self.n_groups,),
            deformed_points,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        x_sampled = einops.rearrange(
            x_sampled, "(b g) d ... -> b (g d) ...", b=batch_size
        )
        keys = self.key(x_sampled)
        values = self.value(x_sampled)
        queries, keys, values = map(
            lambda t: einops.rearrange(t, "b (h d) ... -> b h (...) d", h=self.n_heads),
            (queries, keys, values),
        )
        attention = (
            torch.einsum("b h i d, b h j d -> b h i j", queries, keys)
            * self.scaling_factor
        )
        attention = F.softmax(attention, dim=-1)
        result = torch.einsum("b h i j, b h j d -> b h i d", attention, values)
        result = einops.rearrange(
            result, "b h (s x y) d -> b (h d) s x y", s=depth, x=height, y=width
        )
        result = self.out(result)
        return result

    def create_grid_like(self, tensor, dim=0):
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

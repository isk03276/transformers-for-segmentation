import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformers_for_segmentation.common.attention.multi_head_self_attention import (
    MultiHeadSelfAttention,
)


class DeformableAttention(MultiHeadSelfAttention):
    def __init__(self, n_dim: int, n_heads: int, n_groups: int, kernel_size: int = 1):
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

    def forward(self, x):
        batch_size, n_dim, depth, height, width = x.size()
        queries = self.query(x)
        grouped_queries = einops.rearrange(
            queries,
            "b (g c) d h w  -> (b g) c d h w",
            g=self.n_groups,
            c=self.n_group_dim,
        )
        offset = self.offset_net(grouped_queries)
        n_depth, n_height, n_width = offset.shape[2:]
        n_offset_samples = n_depth * n_height * n_width
        offset = einops.rearrange(offset, "b p d h w -> b d h w p")

        ref_points = self.get_ref_points(
            n_depth, n_height, n_width, batch_size, x.dtype, x.device
        )
        pos = (offset + ref_points).tanh()

        x_sampled = F.grid_sample(
            input=einops.rearrange(
                x,
                "b (g gc) d h w -> (b g) gc d h w",
                g=self.n_groups,
                gc=self.n_group_dim,
            ),
            grid=pos[..., (2, 1, 0)],  # z, y, x  -> x, y, z
            mode="bilinear",
            align_corners=True,
        )
        x_sampled = x_sampled.reshape(batch_size, n_dim, 1, 1, n_offset_samples)

        queries = queries.reshape(
            batch_size * self.n_heads, self.n_head_dim, depth * height * width
        )
        keys = self.key(x_sampled).reshape(
            batch_size * self.n_heads, self.n_head_dim, n_offset_samples
        )
        values = self.value(x_sampled).reshape(
            batch_size * self.n_heads, self.n_head_dim, n_offset_samples
        )

        result = self.scaled_dot_proudction(query=queries, key=keys, value=values)
        result = result.reshape(batch_size, n_dim, depth, height, width)
        return result

    @torch.no_grad()
    def get_ref_points(self, d_key, h_key, w_key, batch_size, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, d_key - 0.5, d_key, dtype=dtype, device=device),
            torch.linspace(0.5, h_key - 0.5, h_key, dtype=dtype, device=device),
            torch.linspace(0.5, w_key - 0.5, w_key, dtype=dtype, device=device),
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 2].div_(w_key).mul_(2).sub_(1)
        ref[..., 1].div_(h_key).mul_(2).sub_(1)
        ref[..., 0].div_(d_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(
            batch_size * self.n_groups, -1, -1, -1, -1
        )  # B * g H W 2

        return ref

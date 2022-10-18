import torch
import torch.nn as nn

from utils.image import slice_image_to_patches


class PatchEmbedder(nn.Module):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_patch: int,
        n_dim: int,
        use_static_positional_encoding: bool,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_channel = n_channel
        self.n_patch = n_patch
        self.n_dim = n_dim
        self.use_static_positional_encoding = use_static_positional_encoding

        self.projection = nn.Conv3d(
            in_channels=n_channel,
            out_channels=n_dim,
            kernel_size=n_patch,
            stride=n_patch,
        )
        self.position_embedding = self.define_position_embedding()

    def define_position_embedding(self) -> torch.Tensor:
        n_patch = self.image_size // self.n_patch
        position_embedding = nn.Parameter(
            torch.randn(self.n_dim, n_patch, n_patch, n_patch)
        )
        return position_embedding

    def forward(self, x):
        embs = self.cnn_patch_projection(x)
        if self.use_static_positional_encoding:
            embs += self.position_embedding
        return embs

    def cnn_patch_projection(self, x):
        embs = self.projection(x)
        return embs

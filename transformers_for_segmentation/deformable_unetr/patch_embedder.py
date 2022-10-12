import torch
import torch.nn as nn

from utils.image import slice_image_to_patches


class PatchEmbedder(nn.Module):
    def __init__(
        self, image_size: int, n_channel: int, n_patch: int, n_dim: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_channel = n_channel
        self.n_patch = n_patch
        self.n_dim = n_dim

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
        # embs += self.position_embedding
        return embs

    def ffn_patch_projection(self, x):
        patches = slice_image_to_patches(
            images=x, patch_size=self.n_patch, flatten=True, is_3d_data=self.is_3d_data
        )
        embs = self.projection(patches)
        return embs

    def cnn_patch_projection(self, x):
        embs = self.projection(x)
        # embs = embs.flatten(start_dim=2)
        # embs = embs.transpose(-1, -2)
        return embs

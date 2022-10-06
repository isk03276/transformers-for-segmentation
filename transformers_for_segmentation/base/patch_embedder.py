import torch
import torch.nn as nn

from utils.image import slice_image_to_patches


class BasePatchEmbedder(nn.Module):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_patch: int,
        n_dim: int,
        use_cnn_embedding: bool,
        is_3d_data: bool,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_channel = n_channel
        self.n_patch = n_patch
        self.n_dim = n_dim
        self.use_cnn_embedding = use_cnn_embedding
        self.is_3d_data = is_3d_data

        if self.use_cnn_embedding:
            conv_args = dict(
                in_channels=n_channel,
                out_channels=n_dim,
                kernel_size=n_patch,
                stride=n_patch,
            )
            if self.is_3d_data:
                self.projection = nn.Conv3d(**conv_args)
            else:
                self.projection = nn.Conv2d(**conv_args)
        else:
            self.projection = self.define_ffn_projection()
        self.position_embedding = self.define_position_embedding()

    def define_ffn_projection(self) -> nn.Linear:
        input_dim = None
        if self.is_3d_data:
            input_dim = self.n_patch ** 3
        else:
            input_dim = self.n_channel * self.n_patch ** 2
        return nn.Linear(input_dim, self.n_dim)

    def define_position_embedding(self) -> torch.Tensor:
        if self.is_3d_data:
            input_dim = (self.image_size // self.n_patch) ** 3
        else:
            input_dim = (self.image_size // self.n_patch) ** 2
        position_embedding = nn.Parameter(torch.randn(input_dim, self.n_dim))
        return position_embedding

    def forward(self, x):
        if self.use_cnn_embedding:
            embs = self.cnn_patch_projection(x)
        else:
            embs = self.ffn_patch_projection(x)
        # (1, 1, dim) -> (B, 1, dim )
        embs += self.position_embedding
        return embs

    def ffn_patch_projection(self, x):
        patches = slice_image_to_patches(
            images=x, patch_size=self.n_patch, flatten=True, is_3d_data=self.is_3d_data
        )
        embs = self.projection(patches)
        return embs

    def cnn_patch_projection(self, x):
        embs = self.projection(x)
        embs = embs.flatten(start_dim=2)
        embs = embs.transpose(-1, -2)
        return embs

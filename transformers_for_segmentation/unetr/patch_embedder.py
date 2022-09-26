import torch
import torch.nn as nn

from transformers_for_segmentation.base.patch_embedder import BasePatchEmbedder
from utils.image import slice_image_to_patches


class PatchEmbedder(BasePatchEmbedder):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_patch: int,
        n_dim: int,
        use_cnn_embedding: bool,
        is_3d_data: bool = True,
    ):
        super().__init__(
            image_size=image_size,
            n_channel=n_channel,
            n_patch=n_patch,
            n_dim=n_dim,
            use_cnn_embedding=use_cnn_embedding,
            is_3d_data=is_3d_data,
        )

    def forward(self, x):
        return super().forward(x)

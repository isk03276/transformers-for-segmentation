import torch
import torch.nn as nn

from transformers_for_segmentation.base.model import BaseModel
from transformers_for_segmentation.common.encoder import EncoderBlock
from transformers_for_segmentation.unetr.patch_embedder import PatchEmbedder


class UnetR(BaseModel):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_patch: int,
        n_dim: int,
        n_encoder_blocks: int,
        n_heads: int,
        use_cnn_embedding: bool,
    ):
        super().__init__(
            image_size=image_size,
            n_channel=n_channel,
            n_patch=n_patch,
            n_dim=n_dim,
            n_encoder_blocks=n_encoder_blocks,
            n_heads=n_heads,
            use_cnn_embedding=use_cnn_embedding,
        )

        self.patch_embedder = PatchEmbedder(
            image_size=self.image_size,
            n_channel=self.n_channel,
            n_patch=self.n_patch,
            n_dim=self.n_dim,
            use_cnn_embedding=self.use_cnn_embedding,
        )
        self.encoders = nn.Sequential(
            *[
                EncoderBlock(n_dim=self.n_dim, n_heads=self.n_heads)
                for _ in range(self.n_encoder_blocks)
            ]
        )
        self.segmentation = None

    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.encoders(x)
        return x

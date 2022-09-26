import torch
import torch.nn as nn

from transformers_for_segmentation.base.model import BaseModel
from transformers_for_segmentation.common.encoder import EncoderBlock
from transformers_for_segmentation.unetr.patch_embedder import PatchEmbedder


class UnetR(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.patch_embedder = PatchEmbedder(self.image_size, self.n_channel, self.n_patch, self.n_dim, self.use_cnn_embedding)
        self.encoders = nn.Sequential(*[EncoderBlock(n_dim=self.n_dim, n_heads=self.n_heads) for _ in range(self.n_encoder_blocks)])
        self.segmentation = None
        
    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.encoders(x)
        # x = self.classifier(x)
        return x
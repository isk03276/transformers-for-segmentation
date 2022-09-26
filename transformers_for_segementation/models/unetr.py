import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.common.modules import EncoderBlock
from models.patch_embedders.unetr_patch_embedder import UnetrPatchEmbedder

class UnetR(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.patch_embedder = UnetrPatchEmbedder(self.image_size, self.n_channel, self.n_patch, self.n_dim, self.use_cnn_embedding)
        self.encoders = nn.Sequential(*[EncoderBlock(n_dim=self.n_dim, n_heads=self.n_heads) for _ in range(self.n_encoder_blocks)])
        self.segmentation = None
        
    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.encoders(x)
        # x = self.classifier(x)
        return x
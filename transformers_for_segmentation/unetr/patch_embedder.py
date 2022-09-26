import torch 
import torch.nn as nn

from transformers_for_segmentation.base.patch_embedder import BasePatchEmbedder
from utils.image import slice_image_to_patches


class PatchEmbedder(BasePatchEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        batch_size = x.size(0)
        # (B, C, H, W) -> (B, N, C  * P * P)
        patches = slice_image_to_patches(
            images=x, patch_size=self.n_patch, flatten=True, is_3d_data=True
        )
        embs = self.projection(patches)
        # (1, 1, dim) -> (B, 1, dim )
        class_token = self.class_token.repeat(batch_size, 1, 1)
        embs = torch.cat((embs, class_token), dim=1)
        embs += self.position_embedding
        return embs

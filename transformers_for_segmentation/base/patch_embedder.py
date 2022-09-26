import torch
import torch.nn as nn


class BasePatchEmbedder(nn.Module):
    def __init__(self, image_size: int, n_channel: int, n_patch: int, n_dim: int, use_cnn_embedding):
        super().__init__()
        self.image_size = image_size
        self.n_channel = n_channel
        self.n_patch = n_patch
        self.n_dim = n_dim
        self.use_cnn_embedding = use_cnn_embedding
        
        if self.use_cnn_embedding:
            self.projection = nn.Conv2d(in_channels=n_channel, out_channels=n_dim, kernel_size=n_patch, stride=n_patch)
        else:
            self.projection = nn.Linear(n_channel * n_patch**2, n_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.n_dim))
        self.position_embedding = nn.Parameter(
            torch.randn((self.image_size // self.n_patch) ** 2 + 1, self.n_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        if self.use_cnn_embedding:
            embs = self.cnn_patch_projection(x)
        else:
            embs = self.ffn_patch_projection(x)
        # (1, 1, dim) -> (B, 1, dim )
        class_token = self.class_token.repeat(batch_size, 1, 1)
        embs = torch.cat((embs, class_token), dim=1)
        embs += self.position_embedding
        return embs 
        
    def ffn_patch_projection(self, x):
        raise NotImplementedError
    
    def cnn_patch_projection(self, x):
        raise NotImplementedError
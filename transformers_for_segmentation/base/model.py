import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_seq: int,
        n_patch: int,
        n_dim: int,
        n_encoder_blocks: int,
        n_heads: int,
        use_cnn_embedding: bool,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_channel = n_channel
        self.n_seq = n_seq
        self.n_patch = n_patch
        self.n_dim = n_dim
        self.n_encoder_blocks = n_encoder_blocks
        self.n_heads = n_heads
        self.use_cnn_embedding = use_cnn_embedding

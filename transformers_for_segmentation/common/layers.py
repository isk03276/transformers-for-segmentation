import torch.nn as nn


class MLPBlock(nn.Module):
    """
    MLP block class in transformer encoder.
    """

    def __init__(self, n_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(n_dim, n_dim)
        self.linear2 = nn.Linear(n_dim, n_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return x

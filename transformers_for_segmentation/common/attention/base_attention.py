from abc import abstractmethod
import torch
import torch.nn as nn


class BaseAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_attention_map(self, query: torch.Tensor, key: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def do_attention(self, attention_map: torch.Tensor, value: torch.Tensor):
        raise NotImplementedError
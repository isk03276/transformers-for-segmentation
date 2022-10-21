from abc import abstractmethod

import torch.nn as nn

from utils.config import load_from_yaml


class BaseModel(nn.Module):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_seq: int,
        n_classes: int,
        model_config_file_path: str,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_channel = n_channel
        self.n_seq = n_seq
        self.n_classes = n_classes
        self.configs = load_from_yaml(model_config_file_path)

        self.encoders = nn.ModuleList()

        self.define_encoder()

    @abstractmethod
    def define_encoder(self):
        pass

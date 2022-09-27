from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from transformers_for_segmentation.base.model import BaseModel


class TensorboardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, tag: str, value: Union[float, int, float, torch.Tensor], step: int):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.flush()
        self.writer.close()

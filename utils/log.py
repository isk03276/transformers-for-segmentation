from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from transformers_for_segmentation.base.model import BaseModel


class TensorboardLogger:
    """
    Logger class for using tensorboard.
    """

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, tag: str, value: Union[float, int, float, torch.Tensor], step: int):
        """
        Log scalar data.
        """
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """
        Close logger.
        """
        self.writer.flush()
        self.writer.close()

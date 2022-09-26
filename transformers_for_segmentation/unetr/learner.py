import torch.nn as nn

from transformers_for_segmentation.base.learner import BaseLearner


class Learner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def define_loss_func(self):
        return nn.CrossEntropyLoss()

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from transformers_for_segmentation.base.model import BaseModel


class BaseLearner:
    def __init__(self, model: BaseModel, lr: float = 3e-4):
        self.model = model
        self.loss_func = nn.CrossEntropyLoss() #self.define_loss_func()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

    def define_loss_func(self):
        raise NotImplementedError

    def estimate_loss(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        preds = self.model(images)
        loss = self.loss_func(preds, labels)
        return loss

    def step(
        self, images: torch.Tensor, labels: torch.Tensor, is_train: bool = True
    ) -> Tuple[float, float]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        loss = self.estimate_loss(images, labels)
        if is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

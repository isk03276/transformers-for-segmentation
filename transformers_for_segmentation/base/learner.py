import torch
import torch.nn as nn
import torch.optim as optim

from transformers_for_segmentation.base.model import BaseModel
from utils.evaluate import get_iou, pred_to_image, get_dice


class BaseLearner:
    def __init__(self, model: BaseModel, n_classes: int, lr: float = 3e-4):
        self.model = model
        self.n_classes = n_classes
        self.loss_func = nn.CrossEntropyLoss()  # self.define_loss_func()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

    def define_loss_func(self):
        raise NotImplementedError

    def estimate_loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        loss = self.loss_func(preds, labels)
        return loss

    def step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.BoolTensor,
        is_train: bool = True,
    ) -> dict:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        info_dict = {}

        preds = self.model(images)
        loss = self.estimate_loss(preds, labels)
        info_dict["loss"] = loss.item()

        preds = pred_to_image(preds, class_dim=1)
        dice = get_dice(preds, labels, mask, n_classes=self.n_classes)
        info_dict["dice"] = dice
        info_dict["preds"] = preds

        if is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return info_dict

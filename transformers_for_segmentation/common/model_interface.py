import torch
import torch.nn as nn
import torch.optim as optim

from transformers_for_segmentation.base.model import BaseModel
from utils.evaluate import pred_to_image, get_dice


class ModelInterface:
    """
    Class for managing model training/testing.
    """

    def __init__(self, model: BaseModel, n_classes: int, lr: float = 3e-4):
        self.model = model
        self.n_classes = n_classes
        self.loss_func = self.define_loss_func()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer, T_max=50, eta_min=1e-4
        )

    def define_loss_func(self):
        """
        Define loss function.
        Assume multi-class classification
        """
        return nn.CrossEntropyLoss()

    def estimate_loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Estimate loss.
        """
        loss = self.loss_func(preds, labels)
        return loss

    def step(
        self, images: torch.Tensor, labels: torch.Tensor, is_train: bool = True,
    ) -> dict:
        """
        Do model inference.
        """
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        info_dict = {}

        preds = self.model(images)
        loss = self.estimate_loss(preds, labels)
        info_dict["loss"] = loss.item()

        preds = pred_to_image(preds, class_dim=1)
        dice = get_dice(preds, labels, n_classes=self.n_classes)
        info_dict["dice"] = dice
        info_dict["preds"] = preds

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        return info_dict

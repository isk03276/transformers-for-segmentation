from typing import Union

import numpy as np
import torch

from utils.image import remove_masked_region
from utils.torch import tensor_to_array


def pred_to_image(pred: torch.Tensor, class_dim: int) -> np.ndarray:
    pred_mask = pred.argmax(dim=class_dim)
    image = tensor_to_array(pred_mask)
    return image


def get_dice(
    pred: Union[np.ndarray, torch.Tensor], label, n_classes: int, epsilon: float = 1e-3
):
    """Get dice score. This method has to be validated"""
    if isinstance(pred, torch.Tensor):
        pred = tensor_to_array(pred)
    if isinstance(label, torch.Tensor):
        label = tensor_to_array(label)

    dice_score = 0
    for c in range(n_classes):
        c_in_pred = np.where(pred == c)[0]
        c_in_label = np.where(label == c)[0]
        dice_score += (len(np.where(c_in_pred == c_in_label)[0]) * 2 + epsilon) / (
            len(c_in_pred) + len(c_in_label) + epsilon
        )
    dice_score /= n_classes
    return dice_score

from typing import Union

import numpy as np
import torch

from utils.image import remove_masked_region
from utils.torch import tensor_to_array


def pred_to_image(pred: torch.Tensor, class_dim: int) -> np.ndarray:
    pred_mask = pred.argmax(dim=class_dim)
    image = tensor_to_array(pred_mask)
    return image


def get_iou(
    pred: Union[np.ndarray, torch.Tensor],
    label: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
):
    pred = remove_masked_region(pred, mask)
    label = remove_masked_region(label, mask)

    intersection = np.logical_and(pred, label)
    union = np.logical_or(pred, label)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def get_dice(pred, label, mask, n_classes: int, epsilon: float = 1e-3):
    """Get dice score. This method has to be validated"""
    pred = remove_masked_region(pred, mask)
    label = remove_masked_region(label, mask)

    dice_score = 0
    for c in range(n_classes):
        c_in_pred = np.where(pred == c)[0]
        c_in_label = np.where(label == c)[0]
        dice_score += (len(np.where(c_in_pred == c_in_label)[0]) * 2 + epsilon) / (
            len(c_in_pred) + len(c_in_label) + epsilon
        )
    dice_score /= n_classes
    return dice_score

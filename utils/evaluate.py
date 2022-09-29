from typing import Union

import numpy as np
import torch
from scipy.spatial.distance import dice

from utils.torch import tensor_to_array


def pred_to_image(pred: torch.Tensor, class_dim: int) -> np.ndarray:
    pred_mask = pred.argmax(dim=class_dim)
    image = tensor_to_array(pred_mask)
    return image


def remove_masked_region(
    image: Union[np.ndarray, torch.Tensor], mask: Union[np.ndarray, torch.Tensor]
):
    if isinstance(image, torch.Tensor):
        image = tensor_to_array(image)
    if isinstance(mask, torch.Tensor):
        mask = tensor_to_array(mask)
    return image[mask.astype(bool)]


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


def get_dice(pred, label, mask):
    """Get dice score. This method has to be validated"""
    pred = remove_masked_region(pred, mask)
    label = remove_masked_region(label, mask)

    score = (pred == label).sum() * 2 / (len(pred) + len(label))
    return score

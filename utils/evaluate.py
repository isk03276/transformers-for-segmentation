from typing import Union

import numpy as np
import torch

from utils.torch import tensor_to_array


def pred_to_image(pred: torch.Tensor, class_dim: int) -> np.ndarray:
    pred_mask = pred.argmax(dim=class_dim)
    image = tensor_to_array(pred_mask)
    return image

def get_iou(pred: Union[np.ndarray, torch.Tensor], label: Union[np.ndarray, torch.Tensor]):
    if isinstance(pred, torch.Tensor):
        pred = tensor_to_array(pred)
    if isinstance(label, torch.Tensor):
        label = tensor_to_array(label)
        
    intersection = np.logical_and(pred, label)
    union = np.logical_or(pred, label)
    iou = np.sum(intersection) / np.sum(union)
    return iou

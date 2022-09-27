import numpy as np
import torch
import torch.nn.functional as F

from utils.torch import tensor_to_array


def pred_to_image(pred: torch.Tensor, dim: int) -> np.ndarray:
    pred_mask = pred.argmax(dim=dim)
    image = tensor_to_array(pred_mask)
    print(image.shape)
    return image

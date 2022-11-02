import os
from typing import Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array.
    """
    return tensor.cpu().detach().numpy()


def get_device(device_name: str) -> torch.device:
    """
    Get torch device for using gpu.
    """
    try:
        device = torch.device(device_name)
    except RuntimeError as e:
        print("[Device name error] Use cpu device!")
        device = torch.device("cpu")
    return device


def save_model(model: nn.Module, dir_name: str, file_name: str):
    """
    Save model.
    """
    os.makedirs(dir_name, exist_ok=True)
    dir_name = dir_name.strip()
    if dir_name[-1] != "/":
        dir_name += "/"
    torch.save(model.state_dict(), dir_name + file_name)


def load_model(model: nn.Module, path: str, keywords_to_exclude: Tuple[str] = None):
    """
    Load model.
    """
    model_state_dict = torch.load(path)
    if keywords_to_exclude:
        params_to_exclude = find_params(model_state_dict, keywords_to_exclude)
        for param_to_exclude in params_to_exclude:
            del model_state_dict[param_to_exclude]
    model.load_state_dict(model_state_dict, strict=False)


def freeze_parameters(model: nn.Module, keywords_to_freeze: Tuple[str]):
    """
    Freeze the specific weights in the model.
    """
    if len(keywords_to_freeze) < 1:
        return
    model_state_dict = model.state_dict()
    params_to_freeze = find_params(model_state_dict, keywords_to_freeze)
    for param in params_to_freeze:
        model_state_dict[param].requires_grad = False


def find_params(model_state_dict: OrderedDict, keywords: Tuple[str]):
    """
    Find model parameters as keywords.
    """
    params = []
    for param_name in model_state_dict.keys():
        for keyword in keywords:
            if keyword in param_name:
                params.append(param_name)
    return params

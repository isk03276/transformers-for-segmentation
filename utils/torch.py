import os
from typing import Tuple

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


def load_model(model: nn.Module, path: str, keywards_to_exclude: Tuple[str] = None):
    """
    Load model.
    """
    model_state_dict = torch.load(path)
    if keywards_to_exclude:
        keys_to_exclude = []
        for key in model_state_dict.keys():
            for keyward_to_exclude in keywards_to_exclude:
                if keyward_to_exclude in key:
                    keys_to_exclude.append(key)
        for key_to_exclude in keys_to_exclude:
            del model_state_dict[key_to_exclude]
    model.load_state_dict(model_state_dict, strict=False)

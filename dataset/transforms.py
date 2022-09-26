import numpy as np
import torchvision.transforms as transforms


def get_common_transform(normalize: bool = True, resize_shape: tuple = None):
    transform_compose_list = [transforms.ToTensor()]
    if normalize:
        transform_compose_list.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    if resize_shape:
        transform_compose_list.append(transforms.Resize(resize_shape))
    transform = transforms.Compose(transform_compose_list)
    return transform


def get_btcv_transform():
    transform_compose_list = []
    # transform_compose_list.append(
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # )
    transform_compose_list.append(transforms.ToTensor())
    # transform_compose_list.append(transforms.Resize((98, 98)))
    transform = transforms.Compose(transform_compose_list)
    return transform

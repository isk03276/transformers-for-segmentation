from typing import Union

import numpy as np
import torch
import SimpleITK as sitk

from utils.torch import tensor_to_array


def pad_image(image: Union[np.ndarray, torch.Tensor]):
    pass


def slice_image_to_patches(
    images: torch.Tensor,
    patch_size: int,
    flatten: bool = True,
    is_3d_data: bool = False,
) -> torch.Tensor:
    """
    Split images into patches.
    Assume that images have shape of (N * C * H * W) or (N * C * D * H * W).
    """
    assert isinstance(images, torch.Tensor)

    images_shape = images.shape
    n_channel = images_shape[1]
    if is_3d_data:
        patches = (
            images.unfold(1, n_channel, n_channel)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
            .unfold(4, patch_size, patch_size)
        )
        if flatten:
            patches = patches.flatten(start_dim=1, end_dim=4)
            patches = patches.flatten(start_dim=2)
    else:
        patches = (
            images.unfold(1, n_channel, n_channel)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
            .squeeze(dim=1)
        )
        if flatten:
            patches = patches.flatten(start_dim=1, end_dim=2)
            patches = patches.flatten(start_dim=2)
    return patches


def load_from_nii(
    path_to_load: str, image_size: int, out_spacing: list = [1.5, 1.5, 2.0]
) -> np.ndarray:
    """
    Load image array from the nii file.
    """
    sitk_image = sitk.ReadImage(path_to_load)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()
    out_size = np.array([image_size, image_size, image_size])
    out_spacing = np.array(original_spacing) / (out_size * 1.0 / original_size)
    sitk_image = sitk.Resample(
        sitk_image,
        size=out_size.tolist(),
        outputOrigin=origin,
        outputSpacing=out_spacing,
        outputDirection=direction,
    )
    image = sitk.GetArrayFromImage(sitk_image)
    return image


def channel_padding(image: np.ndarray, n_channel_to_pad: int, channel_axis: int = 0):
    image_shape = list(image.shape)
    image_shape[channel_axis] = n_channel_to_pad
    pad = np.zeros(image_shape)
    return np.concatenate((image, pad), axis=channel_axis)


def remove_masked_region(
    images: Union[np.ndarray, torch.Tensor], masks: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        images = tensor_to_array(images)
    if isinstance(masks, torch.Tensor):
        masks = tensor_to_array(masks)
    return images[masks.astype(bool)]


def remove_padded_channels(
    images: Union[np.ndarray, torch.Tensor], depths: torch.Tensor
) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        images = tensor_to_array(images)
    image_list = []
    for depth, image in zip(depths, images):
        image_list.append(image[:depth, :])
    result = np.array(image_list)
    return result

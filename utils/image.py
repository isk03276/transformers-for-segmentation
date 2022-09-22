from typing import Union, Tuple

import numpy as np
import torch
import SimpleITK as sitk


def pad_image(image: Union[np.ndarray, torch.Tensor]):
    pass


def slice_image_to_patches(
    images: torch.Tensor, patch_size: int, flatten: bool = True, is_3d_data: bool = True,
) -> torch.Tensor:
    """
    Split images into patches.
    Assume that images have shape of (N * C * H * W).
    """
    assert isinstance(images, torch.Tensor)
    assert len(images.shape) == 4

    images_shape = images.shape
    n_batch, n_channel = images_shape[:2]
    if is_3d_data:
        patches = (
            images.unfold(1, patch_size, patch_size)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
        )
        if flatten:
            patches = patches.flatten(start_dim=1, end_dim=3)
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

def load_from_nii(path_to_load: str)-> Tuple[np.ndarray, dict]:
    """
    Load image array from the nii file.
    """
    sitk_image = sitk.ReadImage(path_to_load)
    image = sitk.GetArrayFromImage(sitk_image).astype('float32')
    origin = np.asarray(sitk_image.GetOrigin())
    spacing = np.asarray(sitk_image.GetSpacing())
    direction = np.asarray(sitk_image.GetDirection())
    image_info = {"origin": origin, "spacing": spacing, "direction": direction}
    return image, image_info

def channel_padding(image: np.ndarray, n_channel_to_pad: int, channel_axis: int = 0):
    image_shape = list(image.shape)
    image_shape[channel_axis] = n_channel_to_pad
    pad = np.zeros(image_shape)
    return np.concatenate((image, pad), axis = channel_axis)
    
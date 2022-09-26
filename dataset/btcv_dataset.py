import numpy as np
import torch

from dataset.base_dataset import BaseDataset
from utils.image import load_from_nii, channel_padding


class BTCVDataset(BaseDataset):
    def __init__(self, root: str, transform, max_slices: int = 96, *args, **kwargs):
        super().__init__(root=root, transform=transform, *args, **kwargs)
        self.max_slices = max_slices

    def get_image(self, image_file_name: str):
        image, _ = load_from_nii(image_file_name)
        image = self.image_prepocess(image)
        return image

    def get_label(self, label_file_name: str):
        label, _ = load_from_nii(label_file_name)
        label = self.image_prepocess(label)
        return label

    def image_prepocess(self, image):
        n_channel, _, _ = image.shape
        image = np.resize(image, (n_channel, 96, 96))
        image = channel_padding(image, self.max_slices - n_channel, channel_axis=0)
        image = torch.Tensor(image)
        return image

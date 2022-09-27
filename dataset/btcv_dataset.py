import numpy as np
import torch

from dataset.base_dataset import BaseDataset
from utils.image import load_from_nii, channel_padding


class BTCVDataset(BaseDataset):
    def __init__(self, root: str, transform, max_seq: int = 96, *args, **kwargs):
        super().__init__(root=root, transform=transform, *args, **kwargs)
        self.max_seq = max_seq

    def get_image(self, image_file_name: str):
        image, _ = load_from_nii(image_file_name)
        image = self.image_prepocess(image)
        return image

    def get_label(self, label_file_name: str):
        label, _ = load_from_nii(label_file_name)
        label = self.image_prepocess(label)
        return label

    def image_prepocess(self, image):
        n_seq, _, _ = image.shape
        image = np.resize(image, (n_seq, 96, 96))
        image = channel_padding(image, self.max_seq - n_seq, channel_axis=0)
        image = torch.Tensor(image)
        image = image.unsqueeze(dim=0) # for adding image channel
        return image

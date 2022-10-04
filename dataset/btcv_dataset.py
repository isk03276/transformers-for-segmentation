from typing import Union, Tuple

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
        image, mask, depth = self.image_prepocess(image)
        image = image.unsqueeze(dim=0)  # for adding image channel
        return image, mask, depth

    def get_label(self, label_file_name: str):
        label, _ = load_from_nii(label_file_name)
        label, mask, depth = self.image_prepocess(label)
        label = label.to(dtype=torch.int64)
        return label, mask, depth

    def image_prepocess(self, image) -> Tuple[torch.Tensor, torch.BoolTensor]:
        n_seq, _, _ = image.shape
        image = np.resize(image, (n_seq, 96, 96))
        image = channel_padding(image, self.max_seq - n_seq, channel_axis=0)
        mask = np.zeros_like(image)
        mask[:n_seq, :, :] = 1.0
        mask = torch.BoolTensor(mask)
        image = torch.Tensor(image)
        image[n_seq:, :, :].requires_grad = False
        return image, mask, n_seq

    def __getitem__(self, index: Union[int, torch.Tensor]):
        if torch.is_tensor(index):
            index = index.tolist()
        image_file_name = self.image_files[index]
        label_file_name = self.label_files[index]
        image, images_mask, image_depth = self.get_image(image_file_name)
        label, _, label_depth = self.get_label(label_file_name)
        assert image_depth == label_depth
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label, images_mask, image_depth

from typing import Union, Tuple

import numpy as np
import torch
from dataset.base_dataset import BaseDataset
from utils.image import load_from_nii


class BTCVDataset(BaseDataset):
    def __init__(self, root: str, transform, image_size: int = 96):
        super().__init__(root=root, transform=transform, image_size=image_size)

    def get_image(self, image_file_name: str):
        image = load_from_nii(image_file_name, image_size=self.image_size)
        image = self.image_prepocess(image, is_label=False)
        image = image.unsqueeze(dim=0)  # for adding image channel
        return image

    def get_label(self, label_file_name: str):
        label = load_from_nii(label_file_name, image_size=self.image_size)
        label = self.image_prepocess(label, is_label=True)
        label = label.to(dtype=torch.int64)
        return label

    def image_prepocess(self, image: np.ndarray, is_label: bool) -> torch.Tensor:
        image = torch.Tensor(image)
        if not is_label:
            image = torch.clamp(image, min=-175, max=275)
            image = (image + 175) / (275 + 175)  # min max normalization
        return image

    def __getitem__(self, index: Union[int, torch.Tensor]):
        if torch.is_tensor(index):
            index = index.tolist()
        image_file_name = self.image_files[index]
        label_file_name = self.label_files[index]
        image = self.get_image(image_file_name)
        label = self.get_label(label_file_name)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

import glob
from abc import abstractmethod
from typing import Union

import torch
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose


class BaseDataset(Dataset):
    """
    Base dataset class.
    Assume the following directory arch
        - root
            -images
                -****.nii.gz
            -labels
                -****.nii.gz
    Args:
        root (str): dataset root directory path
        transform (Compose): torchvision transform
        image_size (int): image size (assume height == width == depth)
    """

    def __init__(self, root: str, transform: Compose, image_size: int):
        super().__init__()

        self.image_files = sorted(glob.glob(root + "/images/*"))
        self.label_files = sorted(glob.glob(root + "/labels/*"))
        self.transform = transform
        self.image_size = image_size

    @abstractmethod
    def get_image(self, image_file_name: str) -> torch.Tensor:
        """
        Get images.
        """
        pass

    @abstractmethod
    def get_label(self, label_file_name: str):
        """
        Get labes
        """
        pass

    def __len__(self) -> int:
        """
        Get dataset length.
        """
        num_image_files = len(self.image_files)
        num_label_files = len(self.label_files)
        assert num_image_files == num_label_files
        return num_image_files

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

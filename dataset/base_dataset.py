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

    def __init__(
        self,
        root: str,
        transform: Compose,
        image_size: int,
        testset_ratio: float = None,
    ):
        super().__init__()

        self.transform = transform
        self.image_size = image_size
        self.testset_ratio = testset_ratio

        self.all_image_files = tuple(sorted(glob.glob(root + "/images/*")))
        self.all_label_files = tuple(sorted(glob.glob(root + "/labels/*")))
        self.image_files = None
        self.label_files = None
        self.test_image_files = None
        self.test_label_files = None
        if self.testset_ratio:
            self._split_testset()
        self.init_dataset()

    def init_dataset(self):
        """
        Copy original image files to the image files.
        """
        self.image_files = list(self.all_image_files)
        self.label_files = list(self.all_label_files)

    def set_test_mode(self):
        """
        Copy test image files to the image files.
        """
        self.image_files = list(self.test_image_files)
        self.label_files = list(self.test_label_files)

    def _split_testset(self):
        """
        Split test dataset.
        All image files is copied to spliited all image files
        """
        n_testset = int(len(self.all_image_files) * self.testset_ratio)
        self.test_image_files = self.all_image_files[-n_testset:]
        self.test_label_files = self.all_label_files[-n_testset:]
        self.all_image_files = tuple(list(self.all_image_files)[:-n_testset])
        self.all_label_files = tuple(list(self.all_label_files)[:-n_testset])

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

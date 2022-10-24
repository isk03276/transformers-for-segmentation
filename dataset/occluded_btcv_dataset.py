from typing import Union

import numpy as np
import torch

from dataset.btcv_dataset import BTCVDataset
from utils.image import load_from_nii
from utils.numpy import np_where_for_multi_dim_array


class OccludedBTCVDataset(BTCVDataset):
    """
    Custom occluded BTCV dataset class.
    This dataset covers 3 of 14 classes('8':aorta, '9':inferior vena cava) in the BTCV dataset.
    """

    def __init__(self, root: str, transform, image_size: int = 96):
        super().__init__(root=root, transform=transform, image_size=image_size)
        self.index_occlusion_starting = 40
        self.occlusion_ratio = 0.4
        self.occlusion_location = 0.5

    def get_image(self, image_file_name: str, aorta_index_array, inferior_vena_cava_index_array):
        image = load_from_nii(image_file_name, image_size=self.image_size)
        image = self.make_occlusion(image, aorta_index_array)
        image = self.make_occlusion(image, inferior_vena_cava_index_array)
        image = self.image_prepocess(image, is_label=False)
        image = image.unsqueeze(dim=0)  # for adding image channel
        return image

    def get_label(self, label_file_name: str):
        label = load_from_nii(label_file_name, image_size=self.image_size)
        label[np.where(label<8)] = 0
        label[np.where(label>9)] = 0
        label[np.where(label==8)] = 1
        label[np.where(label==9)] = 2
        aorta_index_array = np_where_for_multi_dim_array(label == 1)
        inferior_vena_cava_index_array = np_where_for_multi_dim_array(label==2)
        label = self.image_prepocess(label, is_label=True)
        label = label.to(dtype=torch.int64)
        return label, aorta_index_array, inferior_vena_cava_index_array

    def image_prepocess(self, image: np.ndarray, is_label: bool) -> torch.Tensor:
        """
        Convert numpy array images to torch tensor and clamp tensor.
        """
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
        label, aorta_index_array, inferior_vena_cava_index_array = self.get_label(label_file_name)
        image = self.get_image(image_file_name, aorta_index_array, inferior_vena_cava_index_array)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
    
    def make_occlusion(self, image, target_index_array):
        target_slices_index = sorted(list(set(target_index_array[:,0])))
        middle_occlusion_location = int(len(target_slices_index) * self.occlusion_location)
        occlusion_half_size = int(len(target_slices_index) * self.occlusion_ratio / 2)
        occlusion_target_slices_index = np.array(target_slices_index[middle_occlusion_location-occlusion_half_size:middle_occlusion_location+occlusion_half_size])
        for slice_idx in range(len(image)):
            if slice_idx in occlusion_target_slices_index:
                image[slice_idx] = np.ones_like(image[slice_idx]) * -1024
        return image

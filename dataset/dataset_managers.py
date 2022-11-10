import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from dataset.common_dataset import CTDataset
from dataset.base_dataset import BaseDataset
from dataset.abdominal.btcv_dataset import BtcvDataset
from dataset.neuro.adam_dataset import AdamDataset


class DatasetGetter:
    """
    Dataset getter class.
    """

    @staticmethod
    def get_dataset_cls(dataset_name: str):
        if dataset_name == "btcv":
            return BtcvDataset
        elif dataset_name == "adam":
            return AdamDataset
        else:
            print("Use Common CT dataset!")
            return CTDataset

    @staticmethod
    def get_dataset(
        dataset_name: str = "btcv",
        path: str = "data/",
        transform=None,
        testset_ratio: float = None,
    ) -> Dataset:
        """
        Get dataset class with the dataset name as input.
        """
        dataset_cls = DatasetGetter.get_dataset_cls(dataset_name=dataset_name)
        dataset = dataset_cls(
            root=path, transform=transform, testset_ratio=testset_ratio
        )
        return dataset

    @staticmethod
    def get_dataset_loader(
        dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        """
        Get dataset loader with the dataset class as input.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


class KFoldManager:
    def __init__(self, dataset: BaseDataset, n_splits: int, shuffle: bool = True):
        self.dataset = dataset
        self.k_fold = (
            KFold(n_splits=n_splits, shuffle=shuffle) if n_splits > 1 else None
        )

    def split_dataset(self) -> list:
        if self.k_fold:
            splits = self.k_fold.split(self.dataset)
        else:
            splits = [[list(range(len(self.dataset))), ()]]
        return splits

    def set_dataset_fold(self, idx_list: list) -> DataLoader:
        self.dataset.image_files = np.array(self.dataset.all_image_files)[idx_list]
        self.dataset.label_files = np.array(self.dataset.all_label_files)[idx_list]

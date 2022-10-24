from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset.btcv_dataset import BTCVDataset
from dataset.occluded_btcv_dataset import OccludedBTCVDataset


class DatasetGetter:
    """
    Dataset getter class.
    """

    @staticmethod
    def get_dataset_cls(dataset_name: str):
        if dataset_name == "btcv":
            return BTCVDataset
        elif dataset_name == "occluded_btcv":
            return OccludedBTCVDataset
        else:
            raise NotImplementedError

    @staticmethod
    def get_dataset(
        dataset_name: str = "btcv", path: str = "data/", transform=None,
    ) -> Dataset:
        """
        Get dataset class with the dataset name as input.
        """
        dataset_cls = DatasetGetter.get_dataset_cls(dataset_name=dataset_name)
        dataset = dataset_cls(root=path, transform=transform)
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

import sys
import unittest

from torch.utils.data import DataLoader
import numpy as np

try:
    from dataset.occluded_btcv_dataset import OccludedBTCVDataset
    from utils.numpy import np_where_for_multi_dim_array
except ModuleNotFoundError:
    sys.path.append(".")
    from dataset.occluded_btcv_dataset import OccludedBTCVDataset
    from utils.numpy import np_where_for_multi_dim_array


class TestOccludedBTCVDataset(unittest.TestCase):
    def test_data_load(self):
        dataset = OccludedBTCVDataset(root="data/btcv/", transform=None, image_size=96)
        dataset_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        
        image, label = next(iter(dataset_loader))

if __name__ == "__main__":
    unittest.main()

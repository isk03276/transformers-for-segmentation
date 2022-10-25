import sys
import unittest

try:
    from dataset.dataset_managers import DatasetGetter, KFoldManager
except ModuleNotFoundError:
    sys.path.append(".")
    from dataset.dataset_managers import DatasetGetter, KFoldManager


class TestDatasetGetter(unittest.TestCase):
    def test_data_load(self):
        dataset = DatasetGetter.get_dataset()
        dataset_loader = DatasetGetter.get_dataset_loader(dataset=dataset, batch_size=2)


if __name__ == "__main__":
    unittest.main()

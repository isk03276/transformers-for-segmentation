import shutil
import sys
import unittest

import torch

try:
    from utils.evaluate import pred_to_image
except ModuleNotFoundError:
    sys.path.append(".")
    from utils.evaluate import pred_to_image


class TestEvalueate(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = 14
        self.n_dim = 1
        self.pred = torch.rand(8, self.n_classes, 96, 96, 96)

    def test_pred_to_image(self):
        image = pred_to_image(self.pred, self.n_dim)
        assert image.shape == (8, 96, 96, 96)


if __name__ == "__main__":
    unittest.main()

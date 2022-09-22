import sys
import unittest

import torch

try:
    from utils.image import slice_image_to_patches
except ModuleNotFoundError:
    sys.path.append(".")
    from utils.image import slice_image_to_patches


class TestImage(unittest.TestCase):
    def test_slice_image_to_patches(self):
        images_2d = torch.zeros((10, 3, 200, 200))
        images_3d = torch.zeros((10, 200, 200, 200))
        patch_size = 2
        patches_2d = slice_image_to_patches(images_2d, patch_size, flatten=True, is_3d_data=False)
        patches_3d = slice_image_to_patches(images_3d, patch_size, flatten=True, is_3d_data=True)
        assert patches_2d.shape == (10, 10000, 3 * patch_size * patch_size)
        assert patches_3d.shape == (10, 1000000, patch_size**3)


if __name__ == "__main__":
    unittest.main()

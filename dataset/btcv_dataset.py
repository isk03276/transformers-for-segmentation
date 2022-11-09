import torch

from dataset.common_dataset import CTDataset


class BTCVDataset(CTDataset):
    """
    BTCV dataset class.
    See https://www.synapse.org/#!Synapse:syn3193805/wiki/217789.
    """

    n_classes = 14

    def __init__(
        self, root: str, transform, image_size: int = 96, testset_ratio: float = None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            image_size=image_size,
            testset_ratio=testset_ratio,
        )
    
    def image_prepocess(self, image: np.ndarray, is_label: bool) -> torch.Tensor:
        """
        Convert numpy array images to torch tensor and clamp tensor.
        """
        image = torch.Tensor(image)
        if not is_label:
            image = torch.clamp(image, min=-175, max=275)
            image = (image + 175) / (275 + 175)  # min max normalization
        return image

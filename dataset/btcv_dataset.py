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

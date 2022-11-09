from dataset.common_dataset import CTDataset


class AdamDataset(CTDataset):
    """
    ADAM dataset class.
    See 'https://adam.isi.uu.nl/data/'.
    """

    n_classes = 3

    def __init__(
        self, root: str, transform, image_size: int = 96, testset_ratio: float = None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            image_size=image_size,
            testset_ratio=testset_ratio,
        )

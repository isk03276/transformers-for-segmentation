import torch
from visdom import Visdom as viz

from utils.image import label_to_rgb


class VisdomMonitor:
    """
    Visdom util class for visualizing images under training/testing in real-time.
    """

    def __init__(self):
        self.visdom = viz()

    def add_train_images(
        self, input_batches: torch.Tensor, label_batches: torch.Tensor
    ):
        """
        Add training images to the visdom server.
        """
        self.add_batched_input_images(
            image_batches=input_batches, caption="Input Image"
        )
        self.add_batched_label_images(
            label_batches=label_batches, caption="Ground Truth"
        )

    def add_batched_input_images(
        self, image_batches: torch.Tensor, caption: str = None
    ):
        """
        Add batched images.
        """
        images = image_batches.squeeze(1).flatten(
            start_dim=0, end_dim=1
        )  # flatten batches
        images = images.unsqueeze(1)  # D*H*W -> D*C*H*W
        self.visdom.images(tensor=images, opts=dict(caption=caption))

    def add_batched_label_images(
        self, label_batches: torch.Tensor, caption: str = None
    ):
        """
        Add label image to the visdom server.
        """
        rgb_labels = label_to_rgb(label_batches)
        images = rgb_labels.squeeze(1).flatten(
            start_dim=0, end_dim=1
        )  # flatten batches
        self.visdom.images(tensor=images, opts=dict(caption=caption))

    def close(self):
        """
        Close visdom.
        """
        self.visdom.close()
        self.server.terminate()

import torch
import numpy as np
from visdom import Visdom as viz

from utils.image import label_to_rgb


class VisdomMonitor:
    def __init__(self):
        self.visdom = viz()

    def add_train_images(
        self, input_batches: torch.Tensor, label_batches: torch.Tensor
    ):
        self.add_batched_input_images(
            image_batches=input_batches, caption="Input Image"
        )
        self.add_batched_label_images(
            label_batches=label_batches, caption="Ground Truth"
        )

    def add_batched_input_images(
        self, image_batches: torch.Tensor, caption: str = None
    ):
        images = image_batches.squeeze().flatten(
            start_dim=0, end_dim=1
        )  # flatten batches
        images = images.unsqueeze(1)  # D*H*W -> D*C*H*W
        self.visdom.images(tensor=images, opts=dict(caption=caption))

    def add_batched_label_images(
        self, label_batches: torch.Tensor, caption: str = None
    ):
        rgb_labels = label_to_rgb(label_batches)
        images = rgb_labels.squeeze().flatten(start_dim=0, end_dim=1)  # flatten batches
        self.visdom.images(tensor=images, opts=dict(caption=caption))

    def close(self):
        self.visdom.close()
        self.server.terminate()

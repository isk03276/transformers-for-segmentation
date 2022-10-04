import subprocess
import time

import torch
import numpy as np
from visdom import Visdom as viz

from utils.image import remove_padded_channels


class VisdomMonitor:
    def __init__(self):
        self.visdom = viz()

    def add_images(
        self, images: torch.Tensor, depths: torch.Tensor, caption: str = None
    ):
        images = remove_padded_channels(images=images.squeeze(), depths=depths)
        for image in images:
            self.visdom.images(
                tensor=np.expand_dims(image, 1), opts=dict(caption=caption)
            )

    def close(self):
        self.visdom.close()
        self.server.terminate()

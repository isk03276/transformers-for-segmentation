import subprocess
import time

import torch
import numpy as np
from visdom import Visdom as viz

from utils.image import remove_padded_channels


class VisdomMonitor:
    def __init__(self):
        self.visdom = viz()

    def add_images(self, images: torch.Tensor, caption: str = None):
        for image in images:
            self.visdom.images(tensor=image, opts=dict(caption=caption))

    def close(self):
        self.visdom.close()
        self.server.terminate()

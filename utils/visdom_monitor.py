import subprocess
import time

import torch
from visdom import server
from visdom import Visdom as viz

from utils.image import remove_masked_region


class VisdomMonitor:
    def __init__(self, ip: str = "localhost", port:int = "8101"):
        self.server = self.create_server(ip=ip, port=port)
        self.visdom = viz()
        
    def create_server(self, ip: str, port: int):
        server = subprocess.Popen("visdom --hostname {} -port {}".format(ip, port), shell=True, stdout=subprocess.DEVNULL)
        time.sleep(5)
        server.communicate()
        return server
        
    def add_images(self, images: torch.Tensor, masks: torch.BoolTensor, caption: str = None):
        images_shape = images.shape
        print(images.shape, masks.shape)
        assert images_shape == masks.shape
        images = remove_masked_region(image=images, mask=masks, flatten=False)
        self.visdom = viz.images(images, opts=dict(caption=caption))
        
    def close(self):
        self.visdom.close()
        self.server.terminate()
        
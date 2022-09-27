import torch.nn as nn


class Deconv3DLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 2, padding: int = 0
    ):
        super().__init__()
        self.dconv_layer = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        return self.dconv_layer(x)


class Conv3dLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
    ):
        super().__init__()
        self.conv_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
    def forward(self, x):
        return self.conv_layer(x)


class Conv3DBlock(nn.Module):
    """
    The conv block in Unetr paper.
    The sizes of (H, W, D) are maintained.
    Args:
        in_channels (int): input channel dimension
        out_channels (int): output channel dimension
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Conv3dLayer(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Deconv3dBlock(nn.Module):
    """
    The deconv block in Unetr paper.
    The sizes of (H, W, D) are halved.
    Args:
        in_channels (int): input channel dimension
        out_channels (int): output channel dimension
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv_layer = Deconv3DLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2
        )
        self.conv_block = Conv3DBlock(
            in_channels=out_channels, out_channels=out_channels, padding=1,
        )

    def forward(self, x):
        x = self.deconv_layer(x)
        x = self.conv_block(x)
        return x

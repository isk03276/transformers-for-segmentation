import torch
import torch.nn as nn

from transformers_for_segmentation.base.model import BaseModel
from transformers_for_segmentation.common.encoder import EncoderBlock
from transformers_for_segmentation.unetr.patch_embedder import PatchEmbedder
from transformers_for_segmentation.unetr.layers import (
    Conv3DBlock,
    Deconv3dBlock,
    Deconv3DLayer,
    Conv3dLayer,
)


class UnetR(BaseModel):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_seq: int,
        n_patch: int,
        n_dim: int,
        n_heads: int,
        use_cnn_embedding: bool,
        n_classes: int,
        n_encoder_blocks: int = 12,
    ):
        super().__init__(
            image_size=image_size,
            n_channel=n_channel,
            n_seq=n_seq,
            n_patch=n_patch,
            n_dim=n_dim,
            n_encoder_blocks=n_encoder_blocks,
            n_heads=n_heads,
            n_classes=n_classes,
            use_cnn_embedding=use_cnn_embedding,
        )

        self.decoding_patch_dim = self.image_size // self.n_patch

        self.patch_embedder = PatchEmbedder(
            image_size=self.image_size,
            n_channel=self.n_channel,
            n_patch=self.n_patch,
            n_dim=self.n_dim,
            use_cnn_embedding=self.use_cnn_embedding,
        )

        self.encoders = nn.ModuleList()
        self.encoders.extend(
            [
                EncoderBlock(n_dim=self.n_dim, n_heads=self.n_heads)
                for _ in range(self.n_encoder_blocks)
            ]
        )

        self.decoder_0 = nn.Sequential(
            Conv3DBlock(in_channels=self.n_channel, out_channels=64),
            Conv3DBlock(in_channels=64, out_channels=64),
        )
        self.decoder_0_merge = nn.Sequential(
            Conv3DBlock(in_channels=128, out_channels=64),
            Conv3DBlock(in_channels=64, out_channels=64),
        )
        self.decoder_output = Conv3dLayer(
            in_channels=64, out_channels=self.n_classes, kernel_size=1
        )

        self.decoder_3 = nn.Sequential(
            Deconv3dBlock(in_channels=self.n_dim, out_channels=128),
            Deconv3dBlock(in_channels=128, out_channels=128),
            Deconv3dBlock(in_channels=128, out_channels=128),
        )
        self.decoder_3_merge = nn.Sequential(
            Conv3DBlock(in_channels=256, out_channels=128),
            Conv3DBlock(in_channels=128, out_channels=128),
        )
        self.decoder_3_upsample = Deconv3DLayer(
            in_channels=128, out_channels=64, kernel_size=2
        )

        self.decoder_6 = nn.Sequential(
            Deconv3dBlock(in_channels=self.n_dim, out_channels=256),
            Deconv3dBlock(in_channels=256, out_channels=256),
        )
        self.decoder_6_merge = nn.Sequential(
            Conv3DBlock(in_channels=512, out_channels=256),
            Conv3DBlock(in_channels=256, out_channels=256),
        )
        self.decoder_6_upsample = Deconv3DLayer(
            in_channels=256, out_channels=128, kernel_size=2
        )

        self.decoder_9 = Deconv3dBlock(in_channels=self.n_dim, out_channels=512)
        self.decoder_9_merge = nn.Sequential(
            Conv3DBlock(in_channels=1024, out_channels=512),
            Conv3DBlock(in_channels=512, out_channels=512),
        )
        self.decoder_9_upsample = Deconv3DLayer(
            in_channels=512, out_channels=256, kernel_size=2
        )

        self.decoder_12 = Deconv3DLayer(
            in_channels=self.n_dim, out_channels=512, kernel_size=2
        )

    def forward(self, x):
        z0 = x
        # encoding
        x = self.patch_embedder(x)
        embs = []
        for encoder_block in self.encoders:
            x = encoder_block(x)
            embs.append(self.embedding_to_image(x))

        # decoding
        z3, z6, z9, z12 = embs[2::3]
        x = self.decoder_12(z12)
        x = torch.cat([x, self.decoder_9(z9)], dim=1)
        x = self.decoder_9_merge(x)
        x = self.decoder_9_upsample(x)
        x = torch.cat([x, self.decoder_6(z6)], dim=1)
        x = self.decoder_6_merge(x)
        x = self.decoder_6_upsample(x)
        x = torch.cat([x, self.decoder_3(z3)], dim=1)
        x = self.decoder_3_merge(x)
        x = self.decoder_3_upsample(x)
        x = torch.cat([x, self.decoder_0(z0)], dim=1)
        x = self.decoder_0_merge(x)
        x = self.decoder_output(x)

        return x

    def embedding_to_image(self, x):
        return x.transpose(-1, -2,).reshape(
            -1,
            self.n_dim,
            self.decoding_patch_dim,
            self.decoding_patch_dim,
            self.decoding_patch_dim,
        )

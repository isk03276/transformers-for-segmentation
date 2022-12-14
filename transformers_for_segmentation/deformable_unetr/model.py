import torch
import torch.nn as nn

from transformers_for_segmentation.unetr.model import UnetR
from transformers_for_segmentation.common.attention.deformable_attention import (
    DeformableAttention,
)
from transformers_for_segmentation.deformable_unetr.patch_embedder import PatchEmbedder


class DeformableUnetR(UnetR):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_seq: int,
        n_classes: int,
        model_config_file_path: str = "configs/deformable_unetr/default.yaml",
    ):
        super().__init__(
            image_size=image_size,
            n_channel=n_channel,
            n_seq=n_seq,
            n_classes=n_classes,
            model_config_file_path=model_config_file_path,
        )

        self.patch_embedder = PatchEmbedder(
            image_size=self.image_size,
            n_channel=self.n_channel,
            n_patch=self.configs["n_patch"],
            n_dim=self.configs["n_dim"],
            use_static_positional_encoding=self.configs[
                "use_static_positional_encoding"
            ],
        )

    def define_encoder(self):
        self.encoders.extend(
            [
                DeformableAttention(
                    n_dim=self.configs["n_dim"],
                    n_heads=self.configs["n_heads"],
                    n_groups=self.configs["n_groups"],
                    use_dynamic_positional_encoding=self.configs[
                        "use_dynamic_positional_encoding"
                    ],
                )
                for _ in range(self.configs["n_encoder_blocks"])
            ]
        )

    def forward(self, x):
        z0 = x
        # # encoding
        x = self.patch_embedder(x)
        embs = []
        for encoder_block in self.encoders:
            x = encoder_block(x)
            embs.append(x)

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

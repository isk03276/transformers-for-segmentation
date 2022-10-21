import torch.nn as nn

from transformers_for_segmentation.deformable_unetr.model import DeformableUnetR
from transformers_for_segmentation.common.attention.patch_wise_deformable_attention import (
    PatchWiseDeformableAttention,
)


class PatchWiseDeformableUnetR(DeformableUnetR):
    def __init__(
        self,
        image_size: int,
        n_channel: int,
        n_seq: int,
        n_classes: int,
        model_config_file_path: str = "configs/patch_wise_deformable_unetr/default.yaml",
    ):
        super().__init__(
            image_size=image_size,
            n_channel=n_channel,
            n_seq=n_seq,
            n_classes=n_classes,
            model_config_file_path=model_config_file_path,
        )

    def define_encoder(self):
        self.encoders.extend(
            [
                PatchWiseDeformableAttention(
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

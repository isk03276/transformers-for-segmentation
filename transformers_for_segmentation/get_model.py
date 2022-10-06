from transformers_for_segmentation.deformable_unetr.model import DeformableUnetR
from transformers_for_segmentation.base.model import BaseModel
from transformers_for_segmentation.unetr.model import UnetR

def get_model(model_name: str) -> BaseModel:
    if model_name == "unetr":
        return UnetR
    elif model_name =="deformable_unetr":
        return DeformableUnetR
    else:
        raise NotImplementedError
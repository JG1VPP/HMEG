import torch
import torch.nn as nn
from mmengine import MODELS
from mmengine.model import BaseModel
from mmengine import build_from_cfg


@MODELS.register_module()
class GAN(BaseModel):
    def __init__(self, gen, img, obj):
        super().__init__()
        self.gen = build_from_cfg(gen, MODELS)
        self.img = build_from_cfg(img, MODELS)
        self.obj = build_from_cfg(obj, MODELS)

    def forward(self, batch):
        return self.gen(batch)

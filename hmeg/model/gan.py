from mmengine import MODELS, build_from_cfg
from mmengine.model import BaseModel

from hmeg.model.losses import LOSSES


@MODELS.register_module()
class GAN(BaseModel):
    def __init__(self, gen, img, obj, gen_loss, img_loss, obj_loss):
        super().__init__()
        self.gen = build_from_cfg(gen, MODELS)
        self.img = build_from_cfg(img, MODELS)
        self.obj = build_from_cfg(obj, MODELS)

        self.gen_loss = build_from_cfg(gen_loss, LOSSES)
        self.img_loss = build_from_cfg(img_loss, LOSSES)
        self.obj_loss = build_from_cfg(obj_loss, LOSSES)

    def forward(self, batch):
        return self.gen(batch)

    def train_step(self, batch, optim_wrapper):
        print("batch", batch)
        print("optims", optim_wrapper)

        import sys

        sys.exit(0)

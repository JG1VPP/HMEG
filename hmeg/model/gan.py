from mmengine import MODELS
from mmengine.model import BaseModel


@MODELS.register_module()
class GAN(BaseModel):
    def __init__(
        self,
        gen,
        img,
        obj,
        loss_gen,
        loss_img,
        loss_obj,
    ):
        super().__init__()

        # modules
        self.gen = MODELS.build(gen)
        self.img = MODELS.build(img)
        self.obj = MODELS.build(obj)

        # losses
        self.loss_gen = MODELS.build(loss_gen)
        self.loss_img = MODELS.build(loss_img)
        self.loss_obj = MODELS.build(loss_obj)

    def forward(self, batch):
        return self.gen(batch)

    def train_step(self, batch, optim_wrapper):
        print("batch", batch)
        print("optims", optim_wrapper)

        import sys

        sys.exit(0)

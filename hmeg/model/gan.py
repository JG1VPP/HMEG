import torch
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
        opt_gen = optim_wrapper["gen"]
        opt_img = optim_wrapper["img"]
        opt_obj = optim_wrapper["obj"]

        loss = {}

        batch = {k: b.cuda() for k, b in batch.items()}

        with opt_img.optim_context(self.img):
            loss.update(self.train_img(opt_img, **batch))

        with opt_obj.optim_context(self.obj):
            loss.update(self.train_obj(opt_obj, **batch))

        with opt_gen.optim_context(self.gen):
            loss.update(self.train_gen(opt_gen, **batch))

        return loss

    def train_img(self, optim, imgs, **kwargs):
        with torch.no_grad():
            fake, bbox, mask, scores, layout = self.gen(**kwargs)

        # torch.Size([8, 3, 256, 256]) torch.Size([1, 3, 256, 256]) torch.Size([71, 4]) torch.Size([71, 16, 16]) torch.Size([983, 9]) torch.Size([71, 1, 64, 64])
        print(imgs.shape, fake.shape, bbox.shape, mask.shape, scores.shape, layout.shape)

        scores_fake = self.img(fake)
        scores_real = self.img(imgs)
        
        # torch.Size([1, 3, 256, 256]) torch.Size([8, 3, 256, 256]) torch.Size([1, 256, 30, 30]) torch.Size([8, 256, 30, 30])
        print(fake.shape, imgs.shape, scores_fake.shape, scores_real.shape)

        loss = dict(
            img=self.loss_img(scores_real, scores_fake),
        )

        optim.update_params(loss)
        return loss

    def train_obj(self, optim, imgs, **kwargs):
        with torch.no_grad():
            fake, bbox, mask, scores, layout = self.gen(**kwargs)

        scores_fake, ac_loss_fake = self.obj(fake, objs, bbox, obj_to_img)
        scores_real, ac_loss_real = self.obj(imgs, objs, bbox, obj_to_img)

        loss = dict(
            obj=self.loss_obj(scores_real, scores_fake),
            ac_real=ac_loss_real,
            ac_fake=ac_loss_fake,
        )

        optim.update_params(loss)
        return loss

    def train_gen(self, optim, imgs, **kwargs):
        fake, bbox, mask, scores, layout = self.gen(**kwargs)

        scores_fake_obj, ac_loss_fake = self.obj(fake)
        scores_fake_img = self.img(fake)

        loss = dict(
            gen_obj=self.loss_gen(scores_fake_obj),
            gen_img=self.loss_gen(scores_fake_img),
            gen_ac=ac_loss_fake,
        )

        optim.update_params(loss)
        return loss

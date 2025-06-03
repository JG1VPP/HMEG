import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MODELS, Registry


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def _make_targets(x, y):
    """
    Inputs:
    - x: PyTorch Tensor
    - y: Python scalar

    Outputs:
    - out: PyTorch Variable with same shape and dtype as x, but filled with y
    """
    return torch.full_like(x, y)


@MODELS.register_module()
class gan_g_loss(nn.Module):
    def forward(self, scores_fake):
        """
        Input:
        - scores_fake: Tensor of shape (N,) containing scores for fake samples

        Output:
        - loss: Variable of shape (,) giving GAN generator loss
        """
        if scores_fake.dim() > 1:
            scores_fake = scores_fake.view(-1)
        y_fake = _make_targets(scores_fake, 1)
        return bce_loss(scores_fake, y_fake)


@MODELS.register_module()
class gan_d_loss(nn.Module):
    def forward(self, scores_real, scores_fake):
        """
        Input:
        - scores_real: Tensor of shape (N,) giving scores for real samples
        - scores_fake: Tensor of shape (N,) giving scores for fake samples

        Output:
        - loss: Tensor of shape (,) giving GAN discriminator loss
        """
        assert scores_real.size() == scores_fake.size()
        if scores_real.dim() > 1:
            scores_real = scores_real.view(-1)
            scores_fake = scores_fake.view(-1)
        y_real = _make_targets(scores_real, 1)
        y_fake = _make_targets(scores_fake, 0)
        loss_real = bce_loss(scores_real, y_real)
        loss_fake = bce_loss(scores_fake, y_fake)
        return loss_real + loss_fake


@MODELS.register_module()
class wgan_g_loss(nn.Module):
    def forward(self, scores_fake):
        """
        Input:
        - scores_fake: Tensor of shape (N,) containing scores for fake samples

        Output:
        - loss: Tensor of shape (,) giving WGAN generator loss
        """
        return -scores_fake.mean()


@MODELS.register_module()
class wgan_d_loss(nn.Module):
    def forward(self, scores_real, scores_fake):
        """
        Input:
        - scores_real: Tensor of shape (N,) giving scores for real samples
        - scores_fake: Tensor of shape (N,) giving scores for fake samples

        Output:
        - loss: Tensor of shape (,) giving WGAN discriminator loss
        """
        return scores_fake.mean() - scores_real.mean()


@MODELS.register_module()
class lsgan_g_loss(nn.Module):
    def forward(self, scores_fake):
        if scores_fake.dim() > 1:
            scores_fake = scores_fake.view(-1)
        y_fake = _make_targets(scores_fake, 1)
        return F.mse_loss(scores_fake.sigmoid(), y_fake)


@MODELS.register_module()
class lsgan_d_loss(nn.Module):
    def forward(self, scores_real, scores_fake):
        assert scores_real.size() == scores_fake.size()
        if scores_real.dim() > 1:
            scores_real = scores_real.view(-1)
            scores_fake = scores_fake.view(-1)
        y_real = _make_targets(scores_real, 1)
        y_fake = _make_targets(scores_fake, 0)
        loss_real = F.mse_loss(scores_real.sigmoid(), y_real)
        loss_fake = F.mse_loss(scores_fake.sigmoid(), y_fake)
        return loss_real + loss_fake

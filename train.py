import argparse
import itertools
import json
import math
import os
import random
from collections import defaultdict
from mmengine import Config

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hmeg.data import imagenet_deprocess_batch
from hmeg.data.crohme import CROHME
from hmeg.model.discriminators import AcCropDiscriminator, PatchDiscriminator
from hmeg.model.generator import Sg2ImModel
from hmeg.model.losses import get_gan_losses
from hmeg.model.metrics import jaccard

from mmengine.runner import Runner

if __name__ == "__main__":
    config = Config.fromfile("config.py")

    train = CROHME(
        npy_path="datasets/crohme2019/link_npy",
        img_path="datasets/crohme2019/Train_imgs",
        pipeline=None,
        test_mode=False,
    )

    #    train_dataloader = Runner.build_dataloader(dict(
    #        batch_size=8,
    #        num_workers=8,
    #        persistent_workers=True,
    #        sampler=dict(type='DefaultSampler', shuffle=True),
    #        dataset=train,
    #    ))

    print(len(train))
